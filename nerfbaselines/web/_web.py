import glob
import contextlib
import shutil
import shlex
from typing import Optional, Tuple, cast, Dict, Any
import logging
import subprocess
import copy
import tempfile
import math
import json
import os
from nerfbaselines._constants import RESULTS_REPOSITORY, SUPPLEMENTARY_RESULTS_REPOSITORY, DATASETS_REPOSITORY
from nerfbaselines import viewer
from nerfbaselines import get_supported_datasets
from nerfbaselines.results import DEFAULT_DATASET_ORDER
from nerfbaselines import results
import packaging.version
import urllib.parse
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _get_average(scenes, metric, sign):
    values = [s.get(metric) for s in scenes]
    if not values or any(v is None for v in values):
        return float("inf") * sign
    return sum(values) / len(values)


def _get_dataset_info_from_registry(dataset_id):
    dataset_info = None
    try:
        dataset_info = cast(Dict[str, Any], results.get_dataset_info(dataset_id))
    except RuntimeError as e:
        if "does not have metadata" not in str(e):
            raise e
    return dataset_info


def _format_cell(value, id):
    if id == "psnr":
        if abs(value) == float("inf"):
            return "-"
        return f"{value:.2f}"
    elif id in {"ssim", "lpips", "lpips_vgg"}:
        if abs(value) == float("inf"):
            return "-"
        return f"{value:.3f}"
    elif id == "total_train_time":
        if abs(value) == float("inf"):
            return "N/A"
        time = value
        if time > 3600:
            return f"{math.floor(time / 3600)}h {math.floor(time / 60) % 60}m {math.ceil(time % 60)}s";
        elif time > 60:
            return f"{math.floor(time / 60)}m {math.ceil(time % 60)}s";
        else:
            return f"{math.ceil(time)}s";
    elif id == "gpu_memory":
        if abs(value) == float("inf"):
            return "N/A"
        memory = value
        if memory > 1024:
            return f"{memory / 1024:.2f} GB"
        else:
            return f"{memory:.2f} MB"
    else:
        return value


def _clean_slug(slug):
    return slug.replace("_", "-").replace(":", "-").replace("/", "-")


def _sort_versions(versions, *, reverse=True):
    def _version_tuple(x):
        if x == "latest":
            out = (100000,)
        if x == "dev":
            out = (float("inf"),)
        else:
            out = tuple(int(y) for y in x.split("."))
        return out + (0,) * (5 - len(out))
    return sorted(versions, key=_version_tuple, reverse=reverse)


def _get_method_licenses():
    from nerfbaselines import get_supported_methods, get_method_spec

    implemented_methods = []
    for method in get_supported_methods():
        spec = get_method_spec(method)
        meta = spec.get("metadata", {})
        if ":" in method:
            continue
        if meta.get("licenses"):
            implemented_methods.append({"name": meta.get("name", method), "licenses": meta["licenses"]})
    implemented_methods.sort(key=lambda x: x.get("name", "").lower())
    return implemented_methods




class WebBuilder:
    def __init__(self, data_path, output, datasets=None, include_docs=None, include_demos=False, base_path: str = ""):
        self.data_path = data_path
        self.datasets = datasets
        self.include_docs = include_docs
        self.include_demos = include_demos
        self.base_path = base_path

        self._dataset_demo_params = None
        self._model_demo_params = None
        self._stack = contextlib.ExitStack()
        self._input_path = os.path.dirname(os.path.abspath(__file__))
        self._output = output

        # Cache
        self._raw_data = None
        self._raw_demo_params = None
        self._tmpdir_data = None
        self._tmpdir_suppmat = None
        self._tmpdir_results = None

    def __enter__(self):
        self._stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stack.__exit__(exc_type, exc_value, traceback)

    def _get_suppmat_path(self):
        if self._tmpdir_suppmat is None:
            tmpdir = self._stack.enter_context(tempfile.TemporaryDirectory())
            subprocess.check_call(f"git clone --depth=1 https://{SUPPLEMENTARY_RESULTS_REPOSITORY}".split() + [tmpdir], env={
                "GIT_LFS_SKIP_SMUDGE": "1"})
            self._tmpdir_suppmat = tmpdir
        return self._tmpdir_suppmat

    def _get_results_path(self):
        if self._tmpdir_results is None:
            tmpdir = self._stack.enter_context(tempfile.TemporaryDirectory())
            subprocess.check_call(f"git clone --depth=1 https://{RESULTS_REPOSITORY}".split() + [tmpdir], env={
                "GIT_LFS_SKIP_SMUDGE": "1"})
            self._tmpdir_results = tmpdir
        return self._tmpdir_results

    def _get_data_path(self, include_lfs_files=None):
        if self._tmpdir_data is None:
            tmpdir = self._stack.enter_context(tempfile.TemporaryDirectory())
            subprocess.check_call(f"git clone --depth=1 https://{DATASETS_REPOSITORY}".split() + [tmpdir], env={
                "GIT_LFS_SKIP_SMUDGE": "1"})
            self._tmpdir_data = tmpdir
        tmpdir = self._tmpdir_data
        if include_lfs_files:
            args = "git lfs pull -I".split() + [",".join(include_lfs_files)]
            subprocess.check_call(args, cwd=tmpdir)
        return tmpdir

    def get_raw_data(self):
        if self._raw_data is not None:
            return self._raw_data
        self._raw_scene_data = {}
        datasets = self.datasets
        raw_data = []
        if self.data_path is not None:
            logging.info(f"Loading data from {self.data_path}")
            if datasets is None:
                all_results = results._list_dataset_results(self.data_path, dataset=None, scenes=None, dataset_info=None)
                datasets = list(set(x[1]["dataset"] for x in all_results))
                datasets.sort(key=lambda x: DEFAULT_DATASET_ORDER.index(x) 
                              if x in DEFAULT_DATASET_ORDER 
                              else len(DEFAULT_DATASET_ORDER))
                logging.info("Found datasets: " + ", ".join(datasets))
            source_path = self.data_path
        else:
            logging.info("Loading data from NerfBaselines repository")

            # Load data for all datasets
            if datasets is None:
                datasets = list(get_supported_datasets())
                datasets.sort(key=lambda x: DEFAULT_DATASET_ORDER.index(x) 
                              if x in DEFAULT_DATASET_ORDER 
                              else len(DEFAULT_DATASET_ORDER))
                logging.info("Selected datasets: " + ", ".join(datasets))

            # Clone results repository
            source_path = self._get_results_path()

        for dataset in datasets:
            dataset_info = (_get_dataset_info_from_registry(dataset) or {}).copy()
            scene_results = results._list_dataset_results(source_path, dataset, scenes=None, dataset_info=dataset_info)
            for _, scene_data in scene_results:
                method_id = scene_data["nb_info"]["method"]
                scene_id = scene_data["nb_info"]["dataset_metadata"]["scene"]
                self._raw_scene_data[f"{method_id}/{dataset}/{scene_id}"] = scene_data
            dataset_info = results._compile_dataset_results(
                scene_results, dataset, scenes=None, dataset_info=dataset_info)
            dataset_info["id"] = dataset
            rel_paths = [os.path.relpath(path, source_path) for path, _ in scene_results]
            def replace_ext(p):
                assert p.endswith(".json")
                return p[:-len(".json")] + ".zip"
            if self.data_path is None:
                # We don't have links for local files
                dataset_info["resolved_paths"] = {
                    replace_ext(path): replace_ext(f"https://{RESULTS_REPOSITORY}/resolve/main/{path}")
                    for path in rel_paths
                }
            raw_data.append(dataset_info)
        self._raw_data = raw_data
        return raw_data

    def reload_source_code(self):
        global __name__
        with open(__file__, "r", encoding="utf8") as f:
            code = f.read()
            old_name = __name__
            try:
                __name__ = "__nomain__"
                exec(code, globals(), globals())
                logging.info("Data loading reloaded")
            except Exception as e:
                logging.error(e)
            finally:
                __name__ = old_name
        # Fix this instance with the new code
        cls = type(self)
        for k, v in vars(WebBuilder).items():
            if k == "__dict__": continue
            setattr(cls, k, v)

    def _get_pages_configuration(self):
        configuration = {
            "include_docs": self.include_docs,
        }
        if self.include_docs is not None:
            root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            assert os.path.exists(os.path.join(root_path, "docs")), "docs directory not found"
            assert os.path.exists(os.path.join(root_path, ".git")), ".git directory not found"

            # Furthermore, we copy the latest docs dir to the temp directory.
            root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            configuration["docs_source_repo"] = root_path
        return configuration

    def _load_raw_dataset_demo_params(self):
        raw_data = self.get_raw_data()

        # Get paths to include
        paths = set()
        dataset_params = {}
        for dataset in raw_data:
            for method in dataset["methods"]:
                for scene in method["scenes"].keys():
                    paths.add(f"{dataset['id']}/{scene}-nbv.json")
        data_path = self._get_data_path(paths)
        for fname in glob.glob(os.path.join(data_path, "**/*-nbv.json")):
            with open(fname, "r", encoding="utf8") as f:
                try:
                    params = json.load(f).get("metadata", {})
                    relname = os.path.relpath(fname, data_path)[: -len("-nbv.json")]
                    url = f"https://{DATASETS_REPOSITORY}/resolve/main/{relname}-nbv.json"
                    local_path = "viewer/" + _clean_slug(relname) + ".json" 
                    dataset_params[relname] = url, local_path, params
                    assert params.get("id") is not None, f"Missing id in {fname}"
                    assert params.get("scene") is not None, f"Missing scene in {fname}"
                    assert relname == f"{params['id']}/{params['scene']}", f"Invalid id/scene in {fname}"
                except Exception as e:
                    logging.error(f"Error loading {fname}: {e}")
        return dataset_params

    def get_raw_demo_params(self):
        if self._raw_demo_params is not None:
            return self._raw_demo_params
        raw_data = self.get_raw_data()

        # Load method params
        tmpdir = self._get_suppmat_path()

        mesh_demo_params = {}
        method_demo_params = {}
        # Find all {method}/{dataset}/{scene}_demo/params.json files
        for dataset in raw_data:
            for method in dataset["methods"]:
                for scene in method["scenes"].keys():
                    relname = f"{method['id']}/{dataset['id']}/{scene}"
                    if os.path.exists(os.path.join(tmpdir, f"{method['id']}/{dataset['id']}/{scene}_demo/params.json")):
                        with open(os.path.join(tmpdir, f"{method['id']}/{dataset['id']}/{scene}_demo/params.json"), "r", encoding="utf8") as f:
                            demo_params = json.load(f)
                        url = f"https://{SUPPLEMENTARY_RESULTS_REPOSITORY}/resolve/main/{method['id']}/{dataset['id']}/{scene}_demo/params.json"
                        scene_url = urllib.parse.urljoin(url, demo_params.get("scene_url", ""))
                        demo_params["scene_url"] = scene_url
                        if demo_params.get("scene_url_per_appearance") is not None:
                            scene_url_per_appearance = {
                                k: urllib.parse.urljoin(url, v)
                                for k, v in demo_params["scene_url_per_appearance"].items()
                            }
                            demo_params["scene_url_per_appearance"] = scene_url_per_appearance
                        local_path = "viewer/" + _clean_slug(relname) + ".json" 
                        method_demo_params[f"{method['id']}/{dataset['id']}/{scene}"] = url, local_path, demo_params
                        if demo_params.get("type") == "mesh":
                            mesh_demo_params[f"{method['id']}/{dataset['id']}/{scene}"] = url, local_path, demo_params

                    # Add mesh demo
                    if os.path.exists(os.path.join(tmpdir, f"{method['id']}/{dataset['id']}/{scene}_mesh/mesh.ply")):
                        url = f"https://{SUPPLEMENTARY_RESULTS_REPOSITORY}/resolve/main/{method['id']}/{dataset['id']}/{scene}_mesh/mesh.ply"
                        demo_params = {
                            "type": "mesh",
                            "mesh_url": f"https://{SUPPLEMENTARY_RESULTS_REPOSITORY}/resolve/main/{method['id']}/{dataset['id']}/{scene}_mesh/mesh.ply",
                        }
                        local_path = "viewer/" + _clean_slug(relname) + "-m.json" 
                        mesh_demo_params[f"{method['id']}/{dataset['id']}/{scene}"] = url, local_path, demo_params

        dataset_params = self._load_raw_dataset_demo_params()
        self._raw_demo_params = {
            "mesh": mesh_demo_params,
            "demo": method_demo_params,
            "dataset": dataset_params,
        }
        return self._raw_demo_params

    def _resolve_data_link(self, dataset_data, method, dataset, scene):
        output_artifact = method["scenes"][scene].get("output_artifact")
        if output_artifact is not None:
            if output_artifact.get("link", None) is not None:
                return output_artifact["link"]

        resolved_paths = dataset_data.get("resolved_paths", {})
        return resolved_paths.get(f"{method['id']}/{dataset}/{scene}.zip", None)

    def _get_method_data(self, data, method: str):
        raw_params = self.get_raw_demo_params()
        dataset = data["id"]
        metrics = data["metrics"] + [{"id": "total_train_time"}, {"id": "gpu_memory"}]
        sign_map = {m["id"]: 1 if m["ascending"] else -1 for m in data["metrics"] if "ascending" in m}
        m = next(x for x in data["methods"] if x["id"] == method)
        scenes_map = {s["id"]: s for s in data["scenes"]}

        def _transform_scene(s):
            out = {
                **{m["id"]: "-" for m in metrics},
                **{k: _format_cell(v, k) for k, v in m["scenes"].get(s["id"], {}).items()},
                **scenes_map.get(s["id"], {}),
            }
            if s["id"] in m["scenes"]:
                out["data_link"] = self._resolve_data_link(data, m, dataset, s["id"])
            demo_path = f"{m['id']}/{dataset}/{s['id']}"
            if demo_path in raw_params["demo"]:
                _, local_link, _ = raw_params["demo"][demo_path]
                assert local_link.startswith("viewer/"), f"Invalid local link {local_link}"
                local_link = local_link[len("viewer/"):]
                out["demo_link"] = self.base_path + f"/viewer/?p={local_link}"
            if demo_path in raw_params["mesh"]:
                _, local_link, _ = raw_params["mesh"][demo_path]
                assert local_link.startswith("viewer/"), f"Invalid local link {local_link}"
                local_link = local_link[len("viewer/"):]
                out["mesh_demo_link"] = self.base_path + f"/viewer/?p={local_link}"
            return out

        scenes = [_transform_scene(s) for s in data["scenes"]]

        # Add sorts to scenes
        for k in metrics:
            v = [m["scenes"].get(s["id"], {}).get(k["id"], -float("inf")) for s in data["scenes"]]
            sort = sorted(range(len(v)), key=lambda i: v[i].lower() if isinstance(v[i], str) else v[i])
            for i, j in enumerate(sort):
                scenes[j]["sort"] = scenes[j].get("sort", {})
                scenes[j]["sort"][k["id"]] = i
        sort = sorted(range(len(scenes)), key=lambda i: scenes[i]["name"].lower())
        for i, j in enumerate(sort):
            scenes[j]["sort"] = scenes[j].get("sort", {})
            scenes[j]["sort"]["name"] = i
        
        average = {
            ("_"+mid["id"]):_get_average([m["scenes"].get(s["id"]) or {} for s in data["scenes"]], mid["id"], sign_map.get(mid["id"], 1))
            for mid in metrics
        }
        average.update({
            mid["id"]: _format_cell(_get_average([m["scenes"].get(s["id"]) or {} for s in data["scenes"]], mid["id"], sign_map.get(mid["id"], 1)), mid["id"])
            for mid in metrics
        })

        if "paper_results" in m:
            paper_scenes = {k[len(f"{dataset}/"):]: v for k, v in m["paper_results"].items() if k.startswith(f"{dataset}/")}
            for s in scenes:
                s["paper_results"] = paper_scenes.get(s["id"], {}).copy()
                if "note" in s["paper_results"]:
                    for k in list(s["paper_results"]):
                        if k.endswith("note"):
                            continue
                        if f"{k}_note" not in s["paper_results"]:
                            s["paper_results"][f"{k}_note"] = s["paper_results"]["note"]

            average_paper_results = {}
            if len(paper_scenes) == len(data["scenes"]):
                for mid in metrics:
                    values = [s["paper_results"].get(mid["id"], None) for s in scenes]
                    if any(v is None for v in values):
                        continue
                    average_paper_results[mid["id"]] = _format_cell(sum(values) / len(values), mid["id"])

            notes = {}
            for scene in paper_scenes.values():
                for k, v in scene.items():
                    if not k.endswith("note"):
                        continue
                    if k not in notes:
                        notes[k] = v
                    elif notes[k] != v:
                        notes[k] = None
            notes = {k: v for k, v in notes.items() if v is not None}
            if "note" in notes:
                for k in average_paper_results:
                    if k.endswith("note"):
                        continue
                    if f"{k}_note" not in list(notes):
                        notes[f"{k}_note"] = notes["note"]
            average_paper_results.update(notes)
            average["paper_results"] = average_paper_results
        out = {
            **m,
            "slug": _clean_slug(m["id"]),
            "average": average, 
            "scenes": scenes,
        }
        return out

    def get_dataset_data(self, data):
        data = copy.deepcopy(data)
        default_metric = data.get("default_metric") or data["metrics"][0]["id"]
        sign = next((1 if m.get("ascending") else -1 for m in data["metrics"] if m["id"] == default_metric), None)
        if sign is None:
            sign = 1
        data["slug"] = _clean_slug(data["id"])
        data["methods"].sort(key=lambda x: sign * x.get(default_metric, -float("inf")))
        data["methods"] = [
            self._get_method_data(data, m["id"])
            for m in data["methods"]
        ]

        # Add sort values to methods
        sort_metrics = []
        max_scenes = max(len(m["scenes"]) for m in data["methods"])
        if data["methods"]:
            sort_metrics = [k[1:] for k in data["methods"][0]["average"] if k.startswith("_")]
        sorts = {m: [method["average"].pop("_"+m, -float("inf")) for method in data["methods"]]
                 for m in sort_metrics}
        sorts["name"] = [m["name"] for m in data["methods"]]
        for k, v in sorts.items():
            sort = sorted(range(len(v)), key=lambda i: v[i].lower() if isinstance(v[i], str) else v[i])
            for j, i in enumerate(sort):
                data["methods"][i]["sort"] = data["methods"][i].get("sort", {})
                data["methods"][i]["sort"][k] = j

                if k != "name":
                    # Update scene sorts
                    for s in data["methods"][i]["scenes"]:
                        s["sort"][k] += max_scenes * j
        return data

    def _get_pages_data(self):
        raw_data = self.get_raw_data()
        datasets = [self.get_dataset_data(rd) for rd in raw_data if rd["methods"]]
        methods_ids = set(x["id"] for d in datasets for x in d["methods"])
        methods = [dict(sum((list(x.items()) for d in datasets for x in d["methods"] if x["id"] == method_id), [])) for method_id in methods_ids]
        methods = [{
            "slug": _clean_slug(method["id"].replace("_", "-")),
            "id": method["id"],
            "name": method["name"],
            "description": method.get("description"),
            "link": method.get("link"),
            "paper_title": method.get("paper_title"),
            "paper_link": method.get("paper_link"),
            "paper_authors": method.get("paper_authors"),
            "datasets": [{
                    **next(m for m in d["methods"] if m["id"] == method["id"]),
                    **{k: v for k, v in d.items() if k not in {"methods", "scenes", "average"}},
                } for d in datasets
                if any(x for x in d["methods"] if x["id"] == method["id"])
            ],
        } for method in methods]
        return {
            "datasets": datasets,
            "methods": methods,
            "method_licenses": _get_method_licenses(),
        }

    def _get_all_routes(self, input_path, data):
        for fname in os.listdir(os.path.join(input_path, "templates")):
            if fname.startswith("_"):
                continue
            if not fname.endswith(".html"):
                continue
            route= f'/{fname.split(".")[0]}'

            # Dataset pages
            if "[dataset]" in route:
                for dataset in data['datasets']:
                    yield (f'/{route.replace("[dataset]", dataset["id"])}', fname, dataset)
                continue

            # Method pages
            if "[method]" in route:
                for method in data["methods"]:
                    yield (f'/{route.replace("[method]", method["id"])}', fname, method)
                continue

            # Index page
            if route == "/index":
                route = "/"
                yield (route, fname, data)
                continue

            # Other pages
            yield (route, fname, {})

    def build_pages(self):
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        logging.info("Generating pages")
        data = self._get_pages_data()
        base_path = self.base_path
        base_path = base_path.strip("/")
        if base_path:
            base_path = f"/{base_path}"
        env = Environment(
            loader=FileSystemLoader(os.path.join(self._input_path, "templates")),
            autoescape=select_autoescape()
        )
        data["has_docs"] = self.include_docs is not None
        for route, template, data in self._get_all_routes(self._input_path, data):
            if route.endswith("/"):
                route += "index"
            os.makedirs(os.path.dirname(self._output + route), exist_ok=True)
            with open(f"{self._output}{route}.html", "w", encoding="utf8") as f:
                template = env.get_template(template)
                f.write(template.render(**data, base_path=base_path))

    def copy_static_files(self):
        logging.info("Copying static files")
        os.makedirs(self._output, exist_ok=True)
        for fname in os.listdir(os.path.join(self._input_path, "public")):
            with open(os.path.join(self._input_path, f"public/{fname}"), "rb") as f:
                with open(f"{self._output}/{fname}", "wb") as f2:
                    f2.write(f.read())

    def build_docs(self):
        if self.include_docs is None:
            return
        logging.info("Building docs")
        configuration = self._get_pages_configuration()
        output = os.path.join(self._output, "docs")

        repo_path = configuration.get("docs_source_repo")
        if repo_path is None:
            raise ValueError("repo_path is required when include_docs is set.")
        # Build docs with sphinx-build
        os.makedirs(output, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{shlex.quote(repo_path)}:{env.get('PYTHONPATH', '')}"
        base_path = f"{self.base_path}/docs"
        current_version = "dev"
        versions = [current_version]
        version_names = [current_version]
        git_versions = []
        if configuration.get("include_docs") == "all":
            # Build all versions (starting with v)
            git_versions = [x[1:] for x in os.popen("git tag").read().split() if x.startswith("v")]
            git_versions = _sort_versions(git_versions)
            versions += git_versions
            version_names += git_versions
            if len(versions) > 1:
                version_names[1] = "latest"
                current_version = git_versions[0]
        else:
            version_names = ["latest"]

        # Build all versions (starting with v)
        for version, version_name in zip(versions, version_names):
            logging.info(f"Building docs for version {version}")
            with tempfile.TemporaryDirectory() as tmpdir:
                if version != "dev":
                    subprocess.check_call(["git", "clone", repo_path, tmpdir])
                    subprocess.check_call(["git", "checkout", f"v{version}"], cwd=tmpdir)
                    subprocess.check_call(["git", "clean", "-f"], cwd=tmpdir)
                    if not os.path.exists(os.path.join(tmpdir, "docs")):
                        logging.warning(f"Version {version} does not have docs. We will only generate API docs.")

                    # Rewrite the old version fixing the old repo path... replacing:
                    # - jkulhanek.com/nerfbaselines -> nerfbaselines.github.io
                    # - [github.com/jkulhanek/nerfbaselines -> github.com/nerfbaselines/nerfbaselines]
                    # - [raw.githubusercontent.com/jkulhanek/nerfbaselines -> raw.githubusercontent.com/nerfbaselines/nerfbaselines]
                    if packaging.version.parse(version) <= packaging.version.parse("1.2.2"):
                        fix_extensions = {".rst", ".md", ".py", ".html"}
                        for root, _, files in os.walk(tmpdir):
                            for file in files:
                                if not any(file.endswith(ext) for ext in fix_extensions):
                                    continue
                                with open(os.path.join(root, file), "r", encoding="utf8") as f:
                                    content = f.read()
                                content = content.replace("jkulhanek.com/nerfbaselines", "nerfbaselines.github.io")
                                content = content.replace("github.com/jkulhanek/nerfbaselines", "github.com/nerfbaselines/nerfbaselines")
                                content = content.replace("raw.githubusercontent.com/jkulhanek/nerfbaselines", "raw.githubusercontent.com/nerfbaselines/nerfbaselines")
                                with open(os.path.join(root, file), "w", encoding="utf8") as f:
                                    f.write(content)

                    # Copy conf.py from the source repo to tempdir/docs/conf.py
                    os.makedirs(os.path.join(tmpdir, "docs"), exist_ok=True)
                    with open(os.path.join(repo_path, "docs", "conf.py"), "r", encoding="utf8") as f, \
                         open(os.path.join(tmpdir, "docs", "conf.py"), "w", encoding="utf8") as f2:
                        f2.write(f.read())

                    # Copy _ext and _templates dirs from the source to tempdir
                    shutil.rmtree(os.path.join(tmpdir, "docs", "_ext"), ignore_errors=True)
                    shutil.copytree(os.path.join(repo_path, "docs", "_ext"), os.path.join(tmpdir, "docs", "_ext"))
                    shutil.rmtree(os.path.join(tmpdir, "docs", "_templates"), ignore_errors=True)
                    shutil.copytree(os.path.join(repo_path, "docs", "_templates"), os.path.join(tmpdir, "docs", "_templates"))
                    shutil.rmtree(os.path.join(tmpdir, "docs", "_static"), ignore_errors=True)
                    shutil.copytree(os.path.join(repo_path, "docs", "_static"), os.path.join(tmpdir, "docs", "_static"))

                    env["PYTHONPATH"] = f"{shlex.quote(tmpdir)}:{env.get('PYTHONPATH', '')}"
                    shutil.rmtree(os.path.join(tmpdir, "docs", "api"), ignore_errors=True)
                    input_path = os.path.join(tmpdir, "docs")
                else:
                    input_path = os.path.join(repo_path, "docs")

                the_output = output
                if version_name != "latest":
                    the_output = os.path.join(output, version)
                subprocess.check_call(["sphinx-build", 
                                       "-D", f"html_context.current_version={version}",
                                       "-D", f"html_context.versions={','.join(versions)}",
                                       "-D", f"html_context.version_names={','.join(version_names)}",
                                       "-D", f'html_context.base_path={base_path}',
                                       "-b", "html", input_path, the_output], env=env)
    def build_viewer(self):
        # Add viewer
        logging.info("Generating viewer")
        viewer_output = os.path.join(self._output, "viewer")
        os.makedirs(viewer_output, exist_ok=True)
        viewer.build_static_viewer(viewer_output, {
            "state": { "disable_public_url": True },
        })

    def build_viewer_params(self):
        raw_params = self.get_raw_demo_params()

        # First, build dataset params
        allpars = sum((
            [(k,) + x for x in v.items()]
            for k, v in raw_params.items()
        ), [])
        for tp, path, (url, local_path, par) in allpars:
            # Process params here
            params = {}
            if tp == "dataset":
                params = viewer.get_viewer_params_from_dataset_metadata(par)
                params["dataset"] = {"url": url}
            else:
                if "sceneUri" in par:
                    par["scene_url"] = urllib.parse.urljoin(url, par["sceneUri"])
                if "meshUri" in par:
                    par["mesh_url"] = urllib.parse.urljoin(url, par["meshUri"])
                if "backgroundColor" in par and (par["backgroundColor"] or "").lower() == "#ffffff":
                    par["background_color"] = [1., 1., 1.]
                if par.get("mesh_url") is not None:
                    par["mesh_url"] = urllib.parse.urljoin(url, par["mesh_url"])
                if par.get("scene_url") is not None:
                    par["scene_url"] = urllib.parse.urljoin(url, par["scene_url"])
                if par.get("scene_url_per_appearance") is not None:
                    par["scene_url_per_appearance"] = {
                        k: urllib.parse.urljoin(url, v)
                        for k, v in par["scene_url_per_appearance"].items()
                    }
                nb_info = self._raw_scene_data[path]['nb_info']
                params = {
                    "renderer": par,
                }
                if path.split("/", 1)[-1] in raw_params["dataset"]:
                    dataset_url, _ , dataset_par = raw_params["dataset"][path.split("/", 1)[-1]]
                    params["dataset"] = {"url": dataset_url}
                    params = viewer.merge_viewer_params(
                        viewer.get_viewer_params_from_nb_info(nb_info),
                        params, 
                        viewer.get_viewer_params_from_dataset_metadata(dataset_par))
                else:
                    logging.warning(f"Missing dataset params for {path.split('/', 1)[-1]}")
                    params = viewer.merge_viewer_params(
                        viewer.get_viewer_params_from_nb_info(nb_info),
                        params
                    )

            # Save params
            fpath = os.path.join(self._output, local_path)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w", encoding="utf8") as f:
                json.dump(params, f, indent=2)

    def build(self):
        self.build_pages()
        self.copy_static_files()
        self.build_viewer()
        self.build_docs()
        self.build_viewer_params()


def build(output: str, 
          data: Optional[str] = None,
          datasets: Optional[Tuple[str, ...]] = None,
          include_docs: Literal["all", "docs", None] = None,
          include_demos: bool = False,
          base_path: str = ""):
    with WebBuilder(data, output, datasets=datasets, 
                    include_docs=include_docs, 
                    include_demos=include_demos,
                    base_path=base_path) as builder:
        builder.build()


def start_dev_server(data: Optional[str] = None,
                     datasets: Optional[Tuple[str, ...]] = None,
                     include_docs: Literal["all", "docs", None] = None,
                     include_demos: bool = False,
                     port: int = 5500):
    from livereload import Server  # type: ignore
    with tempfile.TemporaryDirectory() as output, \
         WebBuilder(data, output, 
                    datasets=datasets,
                    include_docs=include_docs,
                    include_demos=include_demos) as builder:
        # Build first version
        builder.build()
        _data = builder._get_pages_data()
        input_path = builder._input_path

        def _on_dataloading_change():
            try:
                nonlocal _data
                builder.reload_source_code()
                new_data = builder._get_pages_data()
                if json.dumps(_data) != json.dumps(new_data):
                    _data = new_data
                    builder.build_pages()
                    logging.info("Data reloaded")
            except Exception as e:
                logging.error(e)

        # Create server and watch for changes
        server = Server()
        SFH = server.SFH
        class HtmlRewriteSFHserver(SFH):
            async def get(self, path, *args, **kwargs):
                if os.path.exists(os.path.join(output, path.lstrip("/").replace("/", os.sep), "index.html")):
                    if path != "" and not path.endswith("/"):
                        return self.redirect(path + "/")
                    if path == "":
                        page_path = "index.html"
                    else:
                        page_path = path.lstrip("/") + "/index.html"
                    return await super().get(page_path, *args, **kwargs)
                if os.path.exists(os.path.join(output, path + ".html")):
                    return await super().get(path + ".html", *args, **kwargs)
                return await super().get(path, *args, **kwargs)
        server.SFH = HtmlRewriteSFHserver
        logging.getLogger("tornado").setLevel(logging.WARNING)
        server.watch(os.path.join(input_path, "templates/**/*.html"), builder.build_pages)
        server.watch(os.path.join(input_path, "public/**/*"), builder.copy_static_files)
        viewer_path = os.path.dirname(viewer.__file__)
        server.watch(os.path.join(viewer_path, "**/*"), builder.build_viewer)
        server.watch(__file__, _on_dataloading_change)

        configuration = builder._get_pages_configuration()
        docs_source_repo = configuration.get("docs_source_repo")
        if docs_source_repo is not None:
            docs_path = os.path.join(docs_source_repo, "docs")
            def ignore_files(name):
                if name == os.path.join(docs_path, "cli.md"):
                    return True
                if name.startswith(os.path.join(docs_path, "cli.md")):
                    return True
                if name.startswith(os.path.join(docs_path, "_build")):
                    return True
                return False
            server.watch(os.path.join(docs_source_repo, "docs", "**/*"), 
                         builder.build_docs,
                         ignore=ignore_files)
        server._setup_logging = lambda: None
        logging.info("Starting dev server")
        server.serve(root=output, port=port)

