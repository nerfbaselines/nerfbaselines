import sys
import contextlib
import logging
from unittest import mock
import math
from typing import List, Dict, Any, cast, Union, Type, Iterator, Optional, Tuple
import base64
import os
import struct
from pathlib import Path
import json
import warnings
import numpy as np
from .io import open_any
from . import (
    metrics, get_method_spec, 
    get_dataset_spec, 
    get_supported_methods, 
)
from . import DatasetSpecMetadata, LicenseSpec, MethodInfo, Method, MethodSpec
from ._constants import WEBPAGE_URL
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
try:
    from typing import NotRequired, TypedDict
except ImportError:
    from typing_extensions import NotRequired, TypedDict


DEFAULT_DATASET_ORDER = ["mipnerf360", "blender", "tanksandtemples"]
MethodLink = Literal["paper", "website", "results", "none"]


class SceneInfo(TypedDict):
    id: str
    name: str

class MetricInfo(TypedDict):
    id: str
    name: str
    description: str
    ascending: bool
    link: str

class DatasetInfo(TypedDict):
    id: str
    name: str
    description: str
    scenes: List[SceneInfo]
    metrics: List[MetricInfo]
    default_metric: str
    paper_title: str
    paper_authors: List[str]
    paper_link: str
    link: str
    licenses: NotRequired[List[LicenseSpec]]


def get_dataset_info(dataset: str) -> DatasetInfo:
    """
    Get the dataset info from the dataset repository.

    Args:
        dataset: The dataset name (type).

    Returns:
        The dataset info.
    """
    metrics_info_path = Path(metrics.__file__).with_suffix(".json")
    assert metrics_info_path.exists(), f"Metrics info file {metrics_info_path} does not exist"
    metrics_info = json.loads(metrics_info_path.read_text(encoding="utf8"))
    try:
        dataset_info = cast(Optional[DatasetSpecMetadata], get_dataset_spec(dataset).get("metadata", None))
    except RuntimeError as e:
        if "could not find dataset" in str(e).lower():
            warnings.warn(f"Cound not find dataset {dataset}")
            dataset_info = {
                "metrics": ["psnr", "ssim", "lpips"],
                "default_metric": "psnr",
            }
        else:
            raise

    if dataset_info is None:
        warnings.warn(f"Dataset {dataset} does not have metadata")

    # Fill metrics into dataset info
    metrics_dict = {v["id"]: v for v in metrics_info}
    return cast(DatasetInfo, {
        **(dataset_info or {}),
        "metrics": [metrics_dict[v] for v in (dataset_info or {}).get("metrics", [])],
    })


def load_metrics_from_results(results: Dict) -> Dict[str, List[float]]:
    """
    Load the metrics from a results file (obtained from evaluation).

    Args:
        results: A dictionary of results.

    Returns:
        A dictionary containing the metrics.
    """
    out = {}
    metrics_raw = results["metrics_raw"]
    v: str
    for k, v in metrics_raw.items():
        data = base64.b64decode(v)
        values = list(struct.unpack(f"<{len(data)//4}f", data))
        out[k] = values
    if "nb_info" in results and "total_train_time" in results["nb_info"]:
        out["total_train_time"] = results["nb_info"]["total_train_time"]
    if "nb_info" in results and "resources_utilization" in results["nb_info"] and "gpu_memory" in results["nb_info"]["resources_utilization"]:
        out["gpu_memory"] = results["nb_info"]["resources_utilization"]["gpu_memory"]
    return out


@contextlib.contextmanager
def _mock_build_method(spec: MethodSpec) -> Iterator[Type[Method]]:
    from nerfbaselines import backends
    method_implementation = spec.get("method_class", None)
    if method_implementation is None:
        raise RuntimeError(f"Method spec {spec} does not have a method implementation")

    # Resolve method file
    path, _ = method_implementation.split(":")
    if not path.startswith("nerfbaselines.methods."):
        raise RuntimeError(f"Method does not have implementation in the nerfbaselines.methods package: {method_implementation}")

    old_import = __import__
    whitelist = ["nerfbaselines", "numpy", 
                 "io", "sys", "os", "glob", "json", "math",
                 "enum",
                 "types", "contextlib", "warnings", "struct",
                 "importlib", "functools", "unittest", "mock", 
                 "typing", "typing_extensions", "logging", "argparse",
                 "copy", "typeguard", "abc", "tempfile", "_io", "ast",
                 "dataclasses", "shlex", "tokenize", "builtins", 
                 "inspect", "__builtin__",
                 "re", "pickle", "base64", "shutil", "pathlib", "gc", "random",
                 "itertools", "collections", "operator"]
    def _is_blacklisted(name):
        return not any(name.startswith(x+".") or name == x for x in whitelist)
    def _patch_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level > 0:
            return old_import(name, globals, locals, fromlist, level)
        if _is_blacklisted(name):
            return mock.MagicMock()
        return old_import(name, globals, locals, fromlist, level)

    from nerfbaselines._method_utils import _build_method_class_internal
    build_method = _build_method_class_internal
    build_method_name = f"{build_method.__module__}:{build_method.__name__}"
    new_mods = sys.modules.copy()
    for k in list(new_mods.keys()):
        if _is_blacklisted(k):
            del new_mods[k]
    with mock.patch('builtins.__import__', side_effect=_patch_import), \
        mock.patch('importlib.__import__', side_effect=_patch_import), \
        mock.patch('sys.modules', new=new_mods):
        backend_impl = backends.get_backend(spec, "python")
        with backend_impl:
            method_cls = cast(Type[Method], backend_impl.static_call(build_method_name, spec))
            yield method_cls


def get_method_info_from_spec(spec: MethodSpec) -> MethodInfo:
    """
    Get the method info from the method spec without importing the method dependencies.
    If the method info is loaded from the method implementation, the method spec should have the metadata fields.

    Args:
        spec: The method spec.

    Returns:
        The method info.
    """
    supported_camera_models = spec.get("supported_camera_models", None)
    supported_outputs = spec.get("supported_outputs", None)
    required_features = spec.get("required_features", None)
    try:
        with _mock_build_method(spec) as method_cls:
            method_info = method_cls.get_method_info()
            # Validate method info wrt the spec
            if supported_camera_models is not None:
                assert method_info.get("supported_camera_models") == frozenset(supported_camera_models), f"Method {spec['id']} has different supported_camera_models in the method info"
            if required_features is not None:
                assert method_info.get("required_features") == frozenset(required_features), f"Method {spec['id']} has different required_features in the method info"
            if supported_outputs is not None:
                assert method_info.get("supported_outputs") == tuple(supported_outputs), f"Method {spec['id']} has different supported_outputs in the method info"
            return method_info
    except RuntimeError as e:
        if "Method does not have implementation in the nerfbaselines.methods package" in str(e):
            if supported_camera_models is None:
                raise RuntimeError(f"Method {spec['id']} has external implementation and does not have supported_camera_models in the spec")
            if required_features is None:
                raise RuntimeError(f"Method {spec['id']} has external implementation and does not have required_features in the spec")
            if supported_outputs is None:
                raise RuntimeError(f"Method {spec['id']} has external implementation and does not have supported_outputs in the spec")
            return {
                "method_id": spec["id"], 
                "supported_camera_models": frozenset(supported_camera_models),
                "required_features": frozenset(required_features),
                "supported_outputs": tuple(supported_outputs),
            }
        raise


def _is_scene_results(data: Dict) -> bool:
    return (
        ("render_dataset_metadata" in data or
         ("info" in data and "dataset_metadata" in data["info"]) or
         ("nb_info" in data and "dataset_metadata" in data["nb_info"])) and
        "metrics_raw" in data)


def _list_dataset_results(path: Union[str, Path], dataset: Optional[str] = None, scenes: Optional[List[str]] = None, dataset_info=None) -> List[Tuple[str, Dict]]:
    dataset_info = (dataset_info or {}).copy()
    dataset_info_scenes = dataset_info.get("scenes", None)
    if scenes is not None:
        dataset_info_scenes = dataset_info["scenes"] = [next((y for y in dataset_info_scenes if x == y["id"]), dict(id=x)) for x in scenes]
    out = []
    for path in Path(path).glob("**/*.json"):
        try:
            scene_results = json.loads(path.read_text(encoding="utf8"))
        except UnicodeDecodeError as e:
            logging.warning(f"Skipping invalid results file {path}: {e}")
            continue
        except json.JSONDecodeError as e:
            logging.warning(f"Skipping invalid results file {path}: {e}")
            continue
        if not _is_scene_results(scene_results):
            logging.warning(f"Skipping invalid results file {path}")
            continue

        render_dataset_metadata = scene_results.get("render_dataset_metadata", 
                                                    scene_results.get("info", scene_results.get("nb_info", {})).get("dataset_metadata", {}))
        scene_results.setdefault("nb_info", {}).setdefault("dataset_metadata", render_dataset_metadata)
        dataset_ = render_dataset_metadata.get("id", render_dataset_metadata.get("name", None))
        scene_id = render_dataset_metadata.get("scene", None)
        method_id = scene_results.get("nb_info", {}).get("method", None)
        scene_results["dataset"] = dataset_

        # For old data we try to get method from the scene results
        if method_id is None:
            method_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
            logging.warning(f"Method id not found in the results file {path}, using the method id from the path {method_id}")
            scene_results.setdefault("nb_info", {})["method"] = method_id

        if scene_id is None or dataset_ is None or method_id is None:
            logging.warning(f"Skipping results file {path} without render_dataset_metadata (scene, dataset), or nb_info(method_id)")
            continue

        if dataset is not None and dataset_ != dataset:
            logging.debug(f"Skipping results file {path} with dataset {dataset_} != {dataset}")
            continue

        # Skip scenes missing from dataset_info_scenes
        if dataset_info_scenes is not None and not any(x["id"] == scene_id for x in dataset_info_scenes):
            logging.debug(f"Skipping scene {scene_id} not in dataset_info_scenes")
            continue

        out.append((str(path), scene_results))
    return out


def _compile_dataset_results(results_list: List[Tuple[str, Any]], dataset: str, scenes: Optional[List[str]] = None, dataset_info=None) -> Dict[str, Any]:
    """
    Compile list of per-scene results into a dataset results.
    """
    dataset_info = (dataset_info or {}).copy()
    dataset_info_scenes = dataset_info.get("scenes", None)
    if scenes is not None:
        dataset_info_scenes = dataset_info["scenes"] = [next((y for y in dataset_info_scenes if x == y["id"]), dict(id=x)) for x in scenes]
    dataset_info["methods"] = []
    method_data_map = {}
    agg_metrics = {}

    def _add_scene_data(scene_id, method_id, method_data, scene_results, output_artifact=None):
        method_data_ = method_data_map.get(method_id, None)
        if method_data_ is None:
            method_data_ = method_data.copy()
            method_data_["id"] = method_id
            method_data_["scenes"] = {}
            method_data_map[method_id] = method_data_
            dataset_info["methods"].append(method_data_)
        method_data = method_data_

        results = load_metrics_from_results(scene_results)
        if scene_id in method_data_["scenes"]:
            raise RuntimeError(f"Scene {scene_id} already exists for method {method_id}")
        method_data["scenes"][scene_id] = {}
        for k, v in results.items():
            method_data["scenes"][scene_id][k] = round(np.mean(v), 5)
        if output_artifact is not None:
            method_data["scenes"][scene_id]["output_artifact"] = output_artifact

    if isinstance(results_list, (Path, str)):
        results_path = Path(results_list)
        results_list = []
        for path in results_path.glob("**/*.json"):
            scene_results = json.loads(path.read_text(encoding="utf8"))
            results_list.append((str(path), scene_results))

    for results_path, scene_results in results_list:
        render_dataset_metadata = scene_results.get("render_dataset_metadata", 
                                                    scene_results.get("info", scene_results.get("nb_info", {})).get("dataset_metadata", {}))
        scene_id = render_dataset_metadata.get("scene", None)
        method_id = scene_results.get("nb_info", {}).get("method", None)
        method_spec = get_method_spec(method_id)
        method_data = method_spec.get("metadata", {})
        _add_scene_data(scene_id, method_id, method_data, scene_results)

    # Fill the results from the methods registry
    for method_id in get_supported_methods():
        method_spec = get_method_spec(method_id)
        method_data = method_spec.get("metadata", {}).copy()
        method_info = get_method_info_from_spec(method_spec)
        if "supported_camera_models" in method_info:
            method_data["supported_camera_models"] = list(method_info["supported_camera_models"])
        if "required_features" in method_info:
            method_data["required_features"] = list(method_info["required_features"])
        if "supported_outputs" in method_info:
            method_data["supported_outputs"] = list(method_info["supported_outputs"])
        output_artifacts = method_spec.get("output_artifacts", {})
        for key, info in output_artifacts.items():
            if not key.startswith(dataset + "/"):
                continue

            scene_id = key[len(dataset) + 1:]
            # Skip scenes missing from dataset_info_scenes
            if dataset_info_scenes is not None and not any(x["id"] == scene_id for x in dataset_info_scenes):
                continue

            assert info["link"].endswith(".zip"), f"Output artifact link {info['link']} does not end with .zip"
            results_link = info["link"][:-4] + ".json"
            with open_any(results_link, "r") as f:
                results = json.load(f)

            # Only add if the scene is not already in the results
            if method_id not in method_data_map or scene_id not in method_data_map[method_id]["scenes"]:
                _add_scene_data(scene_id, method_id, method_data, results, info)


    # Aggregate the metrics
    if dataset_info_scenes is None:
        warnings.warn("Dataset info does not have scenes, aggregating all scenes from the methods.")
        scene_ids = set()
        for method_data in dataset_info["methods"]:
            scene_ids.update(method_data["scenes"].keys())
        dataset_info_scenes = [{"id": x} for x in sorted(scene_ids)]

    for method_data in dataset_info["methods"]:
        agg_metrics = {}
        for k, scene in method_data["scenes"].items():
            for k, v in scene.items():
                if isinstance(v, dict):
                    continue
                if k not in agg_metrics:
                    agg_metrics[k] = []
                agg_metrics[k].append(v)
        if len(method_data["scenes"]) == len(dataset_info_scenes):
            for k, v in agg_metrics.items():
                method_data[k] = round(np.mean(v), 5)

    # Reorder scenes using the dataset_info_scenes order and add missing data
    for method_data in dataset_info["methods"]:
        method_data["scenes"] = {x["id"]: {**x, **method_data["scenes"][x["id"]]} for x in dataset_info_scenes if x["id"] in method_data["scenes"]}
    return dataset_info


def compile_dataset_results(results_path: Union[Path, str], dataset: str, scenes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compile the results.json file from the results repository.
    """
    results_path = Path(results_path)
    try:
        dataset_info = cast(Dict[str, Any], get_dataset_info(dataset))
    except RuntimeError as e:
        if "does not have metadata" in str(e):
            dataset_info = {}
        else:
            raise e
    scene_results = _list_dataset_results(results_path, dataset, scenes, dataset_info)
    return _compile_dataset_results(scene_results, dataset, scenes, dataset_info)


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    parts = []
    if seconds >= 3600:
        parts.append(f"{int(seconds // 3600):d}h")
        seconds = seconds % 3600
    if seconds >= 60:
        parts.append(f"{int(seconds // 60):d}m")
        seconds = seconds % 60
    parts.append(f"{math.ceil(seconds):d}s")
    return " ".join(parts)


def format_memory(memory: Optional[float]) -> str:
    # Memory is in MB
    if memory is None:
        return "-"
    if memory < 1024:
        return f"{memory:.0f} MB"
    return f"{memory / 1024:.1f} GB"


def _process_data_for_formatting(results, method_links=None):
    columns = ["name"]
    table: List[List[Any]] = [[{"type": "header", "value": "Method"}]]
    align = "l"
    column_orders: List[Optional[bool]] = [None]

    default_metric = results["default_metric"]
    for metric in results["metrics"]:
        columns.append(metric["id"])
        table[-1].append({"type": "header", "value": metric["name"]})
        column_orders.append(metric["ascending"])
        align += "r"

    # Add train time and GPU memory
    columns.append("total_train_time")
    table[-1].append({"type": "header", "value": "Time"})
    column_orders.append(False)
    columns.append("gpu_memory")
    column_orders.append(False)
    table[-1].append({"type": "header", "value": "GPU mem."})
    align += "rr"

    def get(data, path):
        parts = path.split(".")
        for p in parts[:-1]:
            data = data.get(p, {})
        return data.get(parts[-1], None)


    # Add method's data
    header_len = len(table)
    ord_values = [[] for _ in column_orders]
    method_names = []
    default_metric_id = None
    for method in results["methods"]:
        method_names.append(method["name"])
        table.append([])
        for i, (column, asc) in enumerate(zip(columns, column_orders)):
            value = sort_value = get(method, column)
            link = None
            if column == default_metric:
                default_metric_id = i
            if column == "name":
                # Render link if requested
                if method_links == "paper" and "paper_link" in method:
                    link = method['paper_link']
                elif method_links == "website" and "link" in method:
                    link = method['link']
                elif method_links == "results":
                    link = f"{WEBPAGE_URL}/m-{method['id'].replace(':', '--')}"
                sort_value = value.lower()
            elif column == "total_train_time":
                value = format_duration(value)
            elif column == "gpu_memory":
                value = format_memory(value)
            elif isinstance(value, float):
                value = f"{value:.3f}"
                # Round the value for comparisons
                sort_value = float(value)
            elif isinstance(value, int):
                value = f"{value:d}"
            elif value is None:
                value = "-"
            table[-1].append({"type": "value", "value": value, "link": link})
            if asc is None:
                continue
            if sort_value is not None:
                sort_value = -sort_value if asc else sort_value
            else:
                sort_value = float("inf")
            ord_values[i].append(sort_value)

    # Extract 1st,2nd,3rd places
    for i, (asc, ord_vals) in enumerate(zip(column_orders, ord_values)):
        if asc is None:
            continue
        # vals = sorted(list(set(ord_vals)))
        # order_col = [vals.index(x) if x != float("inf") else None for x in ord_vals]
        order_col = [r if math.isfinite(v) else None for r, v in zip(_rank(ord_vals), ord_vals)]
        for j, rank in enumerate(order_col):
            table[j + header_len][i]["rank"] = rank

    # Sort by the default metric
    all_metrics = ",".join(x["id"] for x in results["metrics"])
    if method_names:
        assert default_metric_id is not None, f"Default metric {default_metric} was not found in the set of metrics {all_metrics}."
        order = [x[-1] for x in sorted([(v, method_names[i], i) for i, v in enumerate(ord_values[default_metric_id])])]
        table = table[:1] + [table[i + 1] for i in order]

    return table, align


def _pad_table(table, align):
    if len(table) == 0:
        return table
    cell_lens = [max(len(x[i]) for x in table) for i in range(len(table[0]))]
    def pad(value, align, cell_len):
        padding = (cell_len - len(value)) * " "
        return (value + padding) if align == "l" else (padding + value)
    return [[pad(x[i], align[i], cell_lens[i]) for i in range(len(x))] for x in table]


def _rank(x, invert=False):
    out = [0] * len(x)
    lastval = None
    lasti = None
    for j, (i, val) in enumerate(sorted(list(enumerate(x)), key=lambda x: -x[1] if invert else x[1])):
        if lastval is None or lastval != val:
            lasti = j
        assert lasti is not None
        out[i] = lasti
        lastval = val
    return out


def render_latex_dataset_results_table(results):
    """
    Generates a latex table from the output of the `compile_dataset_results` method.

    Args:
        results: Output of the `nerfbaselines.results.compile_dataset_results` method.
    """
    table_data, align = _process_data_for_formatting(results)
    table = [[x["value"] for x in table_data[0]]]

    def render_cell(value, rank=None, **kwargs):
        del kwargs
        value = str(value)
        if rank == 0:
            return f"\\pf{{{value}}}"
        if rank == 1:
            return f"\\ps{{{value}}}"
        if rank == 2:
            return f"\\pt{{{value}}}"
        return value
    table += [[render_cell(**x) for x in row] for row in table_data[1:]]
    table = _pad_table(table, align)
    return r"""
\providecommand{\pf}[1]{\textbf{#1}}
\providecommand{\ps}[1]{\underline{#1}}
\providecommand{\pt}[1]{#1}
\begin{tabular}{""" + "".join(align) + r"""}\hline
""" + "\n".join(" & ".join(row) + r" \\" for row in table) + r"""
\end{tabular}
"""


def render_markdown_dataset_results_table(results, method_links: MethodLink = "none") -> str:
    """
    Generates a markdown table from the output of the `compile_dataset_results` method.

    Args:
        results: Output of the `nerfbaselines.results.compile_dataset_results` method.
    """
    table_data, align = _process_data_for_formatting(results, method_links=method_links)
    table = [[x["value"] for x in table_data[0]]]

    def render_cell(value, rank=None, link=None, **kwargs):
        del kwargs
        value = str(value)
        if rank == 0:
            value = f"**{value}**"
        if rank == 1:
            value = f"*{value}*"
        if link is not None:
            value = f"[{value}]({link})"
        return value
    table += [[render_cell(**x) for x in row] for row in table_data[1:]]
    table = _pad_table(table, align)

    lines = ["| " + " | ".join(row) + " |" for row in table]
    # Add header separator
    def pad_splitter(align, cell_len):
        cell_len += 2
        value = (cell_len - 1) * "-"
        return f":{value}" if align == "l" else f"{value}:"

    cell_lens = [max(len(x[i]) for x in table) for i in range(len(table[0]))]
    lines.insert(1, "|" + "|".join(map(pad_splitter, align, cell_lens)) + "|")
    return "".join(x+"\n" for x in lines)
