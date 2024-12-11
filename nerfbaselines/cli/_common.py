import time
import logging
import traceback
import sys
from functools import wraps
import click
import gzip
import io
import hashlib
import os
import itertools

from typing import Any, cast
import itertools
import numpy as np
import pprint
import json
from PIL import Image
from nerfbaselines import BackendName
from nerfbaselines.backends import run_on_host
from nerfbaselines.utils import Indices
from nerfbaselines import NB_PREFIX
try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args

# We reimport setup_logging from backends._common such that
# it can be safely used in RPC backend without adding click dependency
from nerfbaselines.backends._common import setup_logging as setup_logging


@run_on_host()
def _load_ingp(data):
    import msgpack
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        data = f.read()
        data = msgpack.unpackb(data)

    # These keys are not preserved, we drop them from the checkpoint
    data["snapshot"]["nerf"]["rgb"]["rays_per_batch"] = 1 << 12
    data["snapshot"]["nerf"]["rgb"]["measured_batch_size_before_compaction"] = 0
    data["snapshot"]["nerf"]["dataset"]["paths"] = [os.path.basename(p) for p in data["snapshot"]["nerf"]["dataset"]["paths"]]
    return data


class ChangesTracker:
    class ChangesTrackerWithPrefix:
        def __init__(self, tracker, path):
            self._tracker = tracker
            self._path = path

        def __getattr__(self, item):
            return getattr(self._tracker, item)

        def add_dict_changes(self, path, obj1, obj2):
            return self._tracker.add_dict_changes(self._path + path, obj1, obj2)

        def add_file_changes(self, path, file1, file2):
            return self._tracker.add_file_changes(self._path + path, file1, file2)

        def add_dir_changes(self, path, path1, path2):
            return self._tracker.add_dir_changes(self._path + path, path1, path2)

    def __init__(self):
        self._changes = {}
        self._only_diff_options = {}
        self._only_diff_token = 1
        self._fs_token = object()

    def set_show_only_diff(self, path, only_diff):
        changes = self._changes
        for p in path:
            if p not in changes:
                changes[p] = {}
        changes[self._only_diff_token] = only_diff

    def _add_changes(self, path, *args):
        changes = self._changes
        for p in path:
            if p not in changes:
                changes[p] = {}
            changes = changes[p]
        changes[None] = args

    def add_dict_changes(self, path, obj1: Any, obj2: Any):
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            out = False
            o1keys = set(obj1.keys())
            for k in itertools.chain(obj1.keys(), (k for k in obj2.keys() if k not in o1keys)):
                if k not in obj1:
                    v = pprint.pformat(obj2[k])
                    self._add_changes(path + (k,), None, v, True)
                    out = True
                    continue
                if k not in obj2:
                    v = pprint.pformat(obj1[k])[:20]
                    self._add_changes(path + (k,), v, None, True)
                    out = True
                    continue
                if self.add_dict_changes(obj1=obj1.get(k, None), obj2=obj2.get(k, None), path=path + (k,)):
                    out = True
            return out
        v1, v2 = pprint.pformat(obj1), pprint.pformat(obj2)
        if type(obj1) != type(obj2):
            self._add_changes(path, v1, v2, True)
            return True
        if isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
            if len(obj1) != len(obj2):
                self._add_changes(path, f"len: {len(obj1)} != {len(obj2)}", True)
                return True
            out = False
            for i, (o1, o2) in enumerate(zip(obj1, obj2)):
                if self.add_dict_changes(path + (f"[{i}]",), o1, o2):
                    out = True
            return out
        if (
            type(obj1).__name__ == "Parameter" or
            type(obj1).__name__ == "Tensor"
        ):
            obj1 = cast(Any, obj1)
            obj2 = cast(Any, obj2)
            if obj1.device != obj2.device:
                self._add_changes(path, f'device: {obj1.device}', f'device: {obj2.device}', True)
                return True
            elif not np.array_equal(obj1.detach().cpu().numpy(), obj2.detach().cpu().numpy()):
                self._add_changes(path, v1, v2, True)
                return True
            return False
        if isinstance(obj1, np.ndarray):
            if obj1.shape != obj2.shape:
                self._add_changes(path, f"shape: {obj1.shape} != {obj2.shape}", True)
                return True
            if obj1.dtype != obj2.dtype:
                self._add_changes(path, f"dtype: {obj1.dtype} != {obj2.dtype}", True)
                return True
            if obj1.dtype == object:
                return self.add_dict_changes(path, 
                                             {f"[{i}]": v for i, v in enumerate(obj1)},
                                             {f"[{i}]": v for i, v in enumerate(obj2)})
            if not obj1.dtype == object:
                change = not np.array_equal(obj1, obj2)
                self._add_changes(path, v1, v1, change)
                return change
        try:
            if obj1 != obj2:
                self._add_changes(path, v1, v2, True)
                if "datetime" in path[-1].lower() or "version" in path[-1].lower():
                    return False
                return True
        except Exception:
            traceback.print_exc()
            self._add_changes(path, f"failed to comp. {v1} = {v2}", True)
            return True

        self._add_changes(path, v1, v2, False)
        return False

    def clear_changes(self, path):
        changes = self._changes
        for p in path:
            if p not in changes:
                changes[p] = {}
            changes = changes[p]
        changes.clear()

    def add_file_changes(self, path, file1, file2):
        txtextensions = (".txt", ".json", ".md", ".yml", ".yaml", ".csv", ".tsv")
        extension = os.path.splitext(path[-1])[-1].lower()
        isbinary = extension not in txtextensions
        if isinstance(file1, str):
            with open(file1, "r" if not isbinary else "rb") as f1:
                return self.add_file_changes(path, f1, file2)
        if isinstance(file2, str):
            with open(file2, "r" if not isbinary else "rb") as f2:
                return self.add_file_changes(path, file1, f2)
        fpath = tuple(path if path is not None else os.path.split(file1.name))
        is_formatting = False
        if isbinary and fpath[-1].endswith(".ckpt") or fpath[-1].endswith(".pth"):
            try:
                import torch
                data1 = torch.load(file1)
                data2 = torch.load(file2)
                if self.add_dict_changes(fpath, data1, data2):
                    return True
            except Exception:
                traceback.print_exc()
                pass

        if isbinary and fpath[-1].endswith(".pkl"):
            import pickle
            data1 = pickle.load(file1)
            data2 = pickle.load(file2)
            if self.add_dict_changes(fpath, data1, data2):
                return True

        if isbinary and fpath[-1].endswith(".ingp"):
            data1 = file1.read()
            data2 = file2.read()
            file1.seek(0)
            file2.seek(0)
            data1 = _load_ingp(data1)
            data2 = _load_ingp(data2)
            if self.add_dict_changes(fpath, data1, data2):
                return True

        if isbinary and fpath[-1].endswith(".npy"):
            data1 = np.load(file1, allow_pickle=True)
            data2 = np.load(file2, allow_pickle=True)
            file1.seek(0)
            file2.seek(0)
            if self.add_dict_changes(fpath, data1, data2):
                return True

        if not isbinary and fpath[-1].endswith(".json"):
            is_formatting = True
            data1 = file1.read()
            data2 = file2.read()
            file1.seek(0)
            file2.seek(0)
            data1 = json.loads(data1)
            data2 = json.loads(data2)
            if self.add_dict_changes(fpath, data1, data2):
                return True
        if not isbinary and fpath[-1].endswith(".yaml") or fpath[-1].endswith(".yml"):
            data1 = file1.read()
            data2 = file2.read()
            file1.seek(0)
            file2.seek(0)
            try:
                import yaml
                class Loader(yaml.SafeLoader):
                    pass
                def generic_constructor(loader: yaml.Loader, tag_prefix, node):
                    if isinstance(node, yaml.MappingNode):
                        return loader.construct_mapping(node)
                    elif isinstance(node, yaml.ScalarNode):
                        scalar = loader.construct_scalar(node)
                        if not scalar and tag_prefix:
                            if tag_prefix.startswith("yaml.org,2002:python/name:"):
                                return tag_prefix[len("yaml.org,2002:python/name:"):]
                            return tag_prefix + scalar
                    elif isinstance(node, yaml.SequenceNode):
                        return loader.construct_sequence(node)
                    else:
                        return None
                Loader.add_multi_constructor('!', generic_constructor)
                Loader.add_multi_constructor('tag:', generic_constructor)
                is_formatting = True
                data1 = yaml.load(data1, Loader=Loader)
                data2 = yaml.load(data2, Loader=Loader)
                if self.add_dict_changes(fpath, data1, data2):
                    return True
            except ImportError:
                pass
        if isbinary:
            both_shas = True
            data1 = data2 = None
            if os.path.exists(file1.name + ".sha256"):
                sha1 = open(file1.name + ".sha256", "r").read().strip()
            elif os.path.exists(file1.name + ".sha"):
                sha1 = open(file1.name + ".sha", "r").read().strip()
            else:
                data1 = file1.read()
                file1.seek(0)
                sha1 = hashlib.sha256(data1).hexdigest()
                both_shas = False
            if os.path.exists(file2.name + ".sha256"):
                sha2 = open(file2.name + ".sha256", "r").read().strip()
            elif os.path.exists(file2.name + ".sha"):
                sha2 = open(file2.name + ".sha", "r").read().strip()
            else:
                data2 = file2.read()
                file2.seek(0)
                sha2 = hashlib.sha256(data2).hexdigest()
                both_shas = False
            change = sha1 != sha2 if both_shas else data1 != data2
            if change and extension in (".jpg", ".jpeg", ".png"):
                im1 = np.array(Image.open(file1)) / 255.
                im2 = np.array(Image.open(file2)) / 255.
                if im1.shape != im2.shape:
                    self._add_changes(fpath, f"shape: {im1.shape} != {im2.shape}", True)
                    return True
                psnr = 10 * np.log10(1 / np.mean((im1 - im2) ** 2))
                self._add_changes(fpath, f"diff. psnr: {psnr:.6f}", change)
            else:
                if change:
                    self.clear_changes(fpath)
                self._add_changes(fpath, "sha:" + sha1, "sha:" + sha2, change)
            return change
        else:
            data1 = file1.read()
            data2 = file2.read()
            file1.seek(0)
            file2.seek(0)
            change = data1 != data2
            self._add_changes(fpath, data1, data2, change)
            if change and is_formatting:
                c = self._changes
                for p in fpath[:-1]:
                    c = c[p]
                c.clear()
                c[None] = (None, "Formatting change", True)
            return change

    def add_dir_changes(self, path, path1, path2):
        if os.path.isfile(path1):
            return self.add_file_changes(path, path1, path2)
        elif os.path.isdir(path1):
            out = False
            items1 = list(os.listdir(path1))
            items2 = list(os.listdir(path2))
            for k in items1 + [k for k in items2 if k not in items1]:
                if k not in items1:
                    self._add_changes(path + (k,), None, self._fs_token, True)
                    out = True
                elif k not in items2:
                    self._add_changes(path + (k,), self._fs_token, None, True)
                    out = True
                else:
                    if self.add_dir_changes(path + (k,), os.path.join(path1, k), os.path.join(path2, k)):
                        out = True
            return out
        else:
            raise RuntimeError(f"Path {path1} is not a file or directory")

    def with_prefix(self, path):
        return ChangesTracker.ChangesTrackerWithPrefix(self, path)

    def format_changes(self, indent=2, _values=None, _offset=0, _only_diff=True):
        if _values is None:
            _values = self._changes
        _only_diff = _values.get(self._only_diff_token, _only_diff)
        out = ""
        skip_count = 0
        for k, v in _values.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, dict) and None in v and not any(isinstance(k, str) for k in v.keys()):
                # Simple value change
                is_change = v[None][-1]
                if not is_change and _only_diff and skip_count > 5:
                    if skip_count == 6:
                        out += " " * (_offset * indent) + "...\n"
                    skip_count += 1
                    continue
                if is_change:
                    skip_count = 0
                if len(v[None]) == 2:
                    out += " " * (_offset * indent) + f"{k}: \033[31m{v[None][0]}\033[0m\n"
                    continue
                old_val, new_val = v[None][:-1]
                if old_val is None:
                    if new_val is self._fs_token:
                        out += " " * (_offset * indent) + f"\033[32m{k}\033[0m\n"
                    else:
                        v = new_val.replace("\n", "\\n")[:40]
                        out += " " * (_offset * indent) + f"\033[32m{k}: {v}\033[0m\n"
                elif new_val is None:
                    if old_val is self._fs_token:
                        out += " " * (_offset * indent) + f"\033[9;31m{k}\033[0m\n"
                    else:
                        v = old_val.replace("\n", "\\n")[:40]
                        out += " " * (_offset * indent) + f"\033[9;31m{k}: {v}\033[0m\n"
                elif is_change:
                    v1 = old_val.replace("\n", "\\n")[:20]
                    v2 = new_val.replace("\n", "\\n")[:20]
                    out += " " * (_offset * indent) + f"{k}: \033[9;31m{v1}\033[32m{v2}\033[0m\n"
                else:
                    v1 = old_val.replace("\n", "\\n")[:40]
                    out += " " * (_offset * indent) + f"\033[90m{k}: {v1}\033[0m\n"
                    skip_count += 1
                continue
            if isinstance(v, dict):
                sub = self.format_changes(indent, v, _offset + 1, _only_diff)
                if sub or not _only_diff:
                    out += " " * (_offset * indent) + f"{k}:\n"
                    out += sub
                continue
        return out


    def print_changes(self, indent=2):
        print(self.format_changes(indent))


class IndicesClickType(click.ParamType):
    name = "indices"

    def convert(self, value, param, ctx):
        del param, ctx
        if value is None:
            return None
        if isinstance(value, Indices):
            return value
        if ":" in value:
            parts = [int(x) if x else None for x in value.split(":")]
            assert len(parts) <= 3, "too many parts in slice"
            return Indices(slice(*parts))
        return Indices([int(x) for x in value.split(",")])


class TupleClickType(click.ParamType):
    name = "comma-separated-tuple"

    def convert(self, value, param, ctx):
        del param, ctx
        if value is None:
            return None
        if isinstance(value, tuple):
            return value
        return tuple(value.split(","))


class SetParamOptionType(click.ParamType):
    name = "key-value"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, tuple):
            return value
        if "=" not in value:
            self.fail(f"expected key=value pair, got {value}", param, ctx)
        k, v = value.split("=", 1)
        return k, v


def _handle_cli_error(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            write_to_logger = getattr(e, "write_to_logger", None)
            if write_to_logger is not None:
                write_to_logger()
                sys.exit(1)
            else:
                raise e

    return wrapped


def click_backend_option():
    all_backends = list(get_args(BackendName))
    return click.option("--backend", "backend_name", type=click.Choice(all_backends), envvar="NERFBASELINES_BACKEND", help="The backend to use. If not specified, a supported installed  backend is selected automatically. Note, the backend can be specified via the NERFBASELINES_BACKEND environment variable.")


def warn_if_newer_version_available():
    if os.environ.get("NERFBASELINES_NO_UPDATE_CHECK", "0") == "1":
        return
    import requests
    from packaging import version
    from nerfbaselines import __version__
    if __version__ in ("dev", "develop"):
        return
    latest_version_str = None
    try:
        with open(os.path.join(NB_PREFIX, ".latest-version-cache"), "r") as f:
            _latest_version_str, update_time_str = f.read().strip().split("\n")
            update_time = float(update_time_str)
        if time.time() - update_time < 3600 * 24:
            latest_version_str = _latest_version_str
    except FileNotFoundError:
        pass

    if latest_version_str is None:
        try:
            r = requests.get("https://pypi.org/pypi/nerfbaselines/json")
            r.raise_for_status()
            latest_version_str = r.json()["info"]["version"]
            try:
                os.makedirs(NB_PREFIX, exist_ok=True)
                with open(os.path.join(NB_PREFIX, ".latest-version-cache"), "w") as f:
                    f.write(f"{latest_version_str}\n{time.time()}")
                logging.debug("Updated latest version cache")
            except Exception as e:
                logging.exception(e)
        except Exception:
            # No network connection
            return
    logging.debug(f"Latest version: {latest_version_str}, current version: {__version__}")

    latest_version = version.parse(latest_version_str)
    current_version = version.parse(__version__)
    if latest_version > current_version:
        logging.warning(f"New version of nerfbaselines available: {latest_version} (current version: {__version__})! Upgrade with `pip install nerfbaselines --upgrade`")


class NerfBaselinesCliCommand(click.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_params(self, ctx):
        rv = list(super().get_params(ctx))
        rv.insert(len(rv)-1, click.Option(("--verbose", "-v"), is_flag=True, help="Enable verbose logging."))
        return rv

    def invoke(self, ctx):
        setup_logging(ctx.params.pop("verbose", False))
        warn_if_newer_version_available()
        return _handle_cli_error(super().invoke)(ctx)
