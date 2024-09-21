import numpy as np
import shutil
import shlex
from typing import Optional, Tuple
from contextlib import contextmanager
import logging
import subprocess
import copy
import tempfile
from functools import partial
import math
import json
import os
from nerfbaselines._constants import RESULTS_REPOSITORY, SUPPLEMENTARY_RESULTS_REPOSITORY
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


def _resolve_data_link(data, method, dataset, scene):
    output_artifact = method["scenes"][scene].get("output_artifact")
    if output_artifact is not None:
        if output_artifact.get("link", None) is not None:
            return output_artifact["link"]

    resolved_paths = data.get("resolved_paths", {})
    link = resolved_paths.get(f"{method['id']}/{dataset}/{scene}.zip", None)
    return link


def _get_method_data(data, method: str):
    dataset = data["id"]
    metrics = data["metrics"] + [{"id": "total_train_time"}, {"id": "gpu_memory"}]
    sign_map = {m["id"]: 1 if m["ascending"] else -1 for m in data["metrics"] if "ascending" in m}
    m = next(x for x in data["methods"] if x["id"] == method)
    scenes_map = {s["id"]: s for s in data["scenes"]}

    scenes = [{
        **{m["id"]: "-" for m in metrics},
        **{k: _format_cell(v, k) for k, v in m["scenes"].get(s["id"], {}).items()},
        **scenes_map.get(s["id"], {}),
        "data_link": (
            _resolve_data_link(data, m, dataset, s["id"]) if s["id"] in m["scenes"]
            else None
        )
    } for s in data["scenes"]]

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
    return {
        **m,
        "slug": _clean_slug(m["id"]),
        "average": average, 
        "scenes": scenes,
    }


def get_dataset_data(raw_data):
    data = copy.deepcopy(raw_data)
    default_metric = data.get("default_metric") or data["metrics"][0]["id"]
    sign = next((1 if m.get("ascending") else -1 for m in data["metrics"] if m["id"] == default_metric), None)
    if sign is None:
        sign = 1
    data["slug"] = _clean_slug(raw_data["id"])
    data["methods"].sort(key=lambda x: sign * x.get(default_metric, -float("inf")))
    data["methods"] = [
        _get_method_data(data, m["id"])
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


def _clean_slug(slug):
    return slug.replace("_", "-").replace(":", "-").replace("/", "-")


def get_raw_data(data_path):
    dataset_ids = [f[:-5] for f in os.listdir(data_path) if f.endswith(".json")]
    return [{"id": dataset_id, **json.load(open(f"{data_path}/{dataset_id}.json", "r", encoding="utf8"))} 
            for dataset_id in dataset_ids]


def get_data(raw_data):
    datasets = [get_dataset_data(rd) for rd in raw_data if rd["methods"]]
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


def get_all_routes(input_path, data):
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


def _generate_pages(data, input_path, output, configuration):
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    base_path = configuration.get("base_path", "")
    base_path = base_path.strip("/")
    if base_path:
        base_path = f"/{base_path}"
    env = Environment(
        loader=FileSystemLoader(os.path.join(input_path, "templates")),
        autoescape=select_autoescape()
    )
    data["has_docs"] = configuration.get("include_docs") is not None
    for route, template, data in get_all_routes(input_path, data):
        if route.endswith("/"):
            route += "index"
        os.makedirs(os.path.dirname(output + route), exist_ok=True)
        with open(f"{output}{route}.html", "w", encoding="utf8") as f:
            template = env.get_template(template)
            f.write(template.render(**data, base_path=base_path))


def _copy_static_files(input_path, output):
    os.makedirs(output, exist_ok=True)
    for fname in os.listdir(os.path.join(input_path, "public")):
        with open(os.path.join(input_path, f"public/{fname}"), "rb") as f:
            with open(f"{output}/{fname}", "wb") as f2:
                f2.write(f.read())


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



def _build_docs(configuration,
                output):
    repo_path = configuration.get("docs_source_repo")
    if repo_path is None:
        raise ValueError("repo_path is required when include_docs is set.")
    # Build docs with sphinx-build
    os.makedirs(output, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{shlex.quote(repo_path)}:{env.get('PYTHONPATH', '')}"
    base_path = f"{configuration.get('base_path', '')}/docs"
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


def _generate_demo_pages(output, configuration):
    # 3dgs demo
    del configuration
    os.makedirs(os.path.join(output, "demos", "3dgs"), exist_ok=True)
    from nerfbaselines.methods._gaussian_splatting_demo import export_generic_demo
    from ._multidemo import make_multidemo
    export_generic_demo(os.path.join(output, "demos", "3dgs"), options={
        'dataset_metadata': {
            'viewer_transform': np.eye(4),
            'viewer_initial_pose': np.eye(4),
        },
        'mock_cors': True,
        'enable_shared_memory': True,
    })
    make_multidemo(os.path.join(output, "demos", "3dgs"))
    os.remove(os.path.join(output, "demos", "3dgs", "params.json"))

    # Mesh demo
    os.makedirs(os.path.join(output, "demos", "mesh"), exist_ok=True)
    from nerfbaselines.methods._mesh_demo import export_generic_demo
    export_generic_demo(os.path.join(output, "demos", "mesh"), options={
        'dataset_metadata': {
            'viewer_transform': np.eye(4),
            'viewer_initial_pose': np.eye(4),
        },
    })
    make_multidemo(os.path.join(output, "demos", "mesh"))
    os.remove(os.path.join(output, "demos", "mesh", "params.json"))


def _build(input_path, output, raw_data, configuration):
    if os.path.exists(output):
        raise FileExistsError(f"Output directory {output} already exists.")

    # Build docs
    if configuration.get("include_docs") is not None:
        logging.info("Building docs")
        _build_docs(configuration, output=os.path.join(output, "docs"))

    # Generate all routes
    logging.info("Generating pages")
    data = get_data(raw_data)
    _generate_pages(data, input_path, output, configuration)

    # Copy static files
    logging.info("Copying static files")
    _copy_static_files(input_path, output)

    # Add demos
    logging.info("Generating demo pages")
    _generate_demo_pages(output, configuration)


def _reload_data_loading():
    global __name__
    if not os.path.exists(__file__):
        logging.warning("Data loading skipped (file not found)")
        return
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


def build(output: str, 
          data: Optional[str] = None,
          datasets: Optional[Tuple[str, ...]] = None,
          include_docs: Literal["all", "docs", None] = None,
          base_path: str = ""):
    input_path = os.path.dirname(os.path.abspath(__file__))
    with _prepare_data(data, datasets, include_docs=include_docs) as (raw_data, configuration):
        configuration["base_path"] = base_path
        _build(input_path, output, raw_data, configuration)


def start_dev_server(data: Optional[str] = None,
                     datasets: Optional[Tuple[str, ...]] = None,
                     include_docs: Literal["all", "docs", None] = None):
    from livereload import Server  # type: ignore
    with _prepare_data(data, datasets, include_docs=include_docs) as (raw_data, configuration), \
            tempfile.TemporaryDirectory() as output:
        input_path = os.path.dirname(os.path.abspath(__file__))
        del data

        # Build first version
        os.rmdir(output)
        _build(input_path, output, raw_data, configuration)
        _data = get_data(raw_data)

        def _on_dataloading_change():
            nonlocal _data
            _reload_data_loading()
            new_data = get_data(raw_data)
            if json.dumps(_data) != json.dumps(new_data):
                _data = new_data
                _generate_pages(_data, input_path, output, configuration)
                logging.info("Data reloaded")

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
        server.watch(os.path.join(input_path, "templates/**/*.html"), lambda: _generate_pages(_data, input_path, output, configuration))
        server.watch(os.path.join(input_path, "public/**/*"), partial(_copy_static_files, input_path, output))
        server.watch(__file__, _on_dataloading_change)

        build_docs = lambda: _build_docs(
            configuration=configuration,
            output=os.path.join(output, "docs"))
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
                         build_docs,
                         ignore=ignore_files)
        server._setup_logging = lambda: None
        logging.info("Starting dev server")
        server.serve(root=output)


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


@contextmanager
def _prepare_data(data_path, datasets=None, include_docs=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        if data_path is not None:
            logging.info(f"Loading data from {data_path}")
            raw_datasets = get_raw_data(data_path)
            if datasets is None:
                from nerfbaselines.results import DEFAULT_DATASET_ORDER
                raw_datasets.sort(key=lambda x: DEFAULT_DATASET_ORDER.index(x["id"]) 
                                  if x in DEFAULT_DATASET_ORDER 
                                  else len(DEFAULT_DATASET_ORDER))
            if datasets is not None:
                raw_datasets_map = {d["id"]: d for d in raw_datasets}
                raw_datasets = [raw_datasets_map[d] for d in datasets]
            raw_data = raw_datasets
        else:
            logging.info("Loading data from NerfBaselines repository")

            # Load data for all datasets
            from nerfbaselines import get_supported_datasets
            from nerfbaselines.results import compile_dataset_results, DEFAULT_DATASET_ORDER

            if datasets is None:
                datasets = list(get_supported_datasets())
                datasets.sort(key=lambda x: DEFAULT_DATASET_ORDER.index(x) 
                              if x in DEFAULT_DATASET_ORDER 
                              else len(DEFAULT_DATASET_ORDER))
                logging.info("Selected datasets: " + ", ".join(datasets))

            raw_data = []

            # Clone results repository
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.check_call(f"git clone --depth=1 https://{RESULTS_REPOSITORY}".split() + [tmpdir], env={"GIT_LFS_SKIP_SMUDGE": "1"})
                # List all paths in tmpdir
                existing_paths = [os.path.relpath(os.path.join(root, file), tmpdir) for root, _, files in os.walk(tmpdir) for file in files]
                resolved_paths = {
                    path: f"https://{RESULTS_REPOSITORY}/resolve/main/{path}"
                    for path in existing_paths
                }
                for dataset in datasets:
                    dataset_info = compile_dataset_results(tmpdir, dataset)
                    dataset_info["id"] = dataset
                    dataset_info["resolved_paths"] = resolved_paths
                    raw_data.append(dataset_info)

            with tempfile.TemporaryDirectory() as tmpdir:
                # Clone supplementary repository
                subprocess.check_call(f"git clone --depth=1 https://{SUPPLEMENTARY_RESULTS_REPOSITORY}".split() + [tmpdir], env={"GIT_LFS_SKIP_SMUDGE": "1"})

                # Find all {method}/{dataset}/{scene}_demo/params.json files
                for dataset in raw_data:
                    for method in dataset["methods"]:
                        for scene, scene_data in method["scenes"].items():
                            if os.path.exists(os.path.join(tmpdir, f"{method['id']}/{dataset['id']}/{scene}_demo/params.json")):
                                with open(os.path.join(tmpdir, f"{method['id']}/{dataset['id']}/{scene}_demo/params.json"), "r", encoding="utf8") as f:
                                    demo_params = json.load(f)
                                base = f"https://{SUPPLEMENTARY_RESULTS_REPOSITORY}/resolve/main/{method['id']}/{dataset['id']}/{scene}_demo/"
                                query = {
                                    "p": base + "params.json",
                                }
                                if "links" in demo_params:
                                    for i, (label, link) in enumerate(demo_params["links"].items()):
                                        # Make link absolute
                                        if not link.startswith("http"):
                                            link = base + link
                                        # Url encode components
                                        query[f"p{i}"] = label
                                        query[f"p{i}v"] = link
                                if demo_params["type"] == "gaussian-splatting":
                                    scene_data["demo_link"] = f"./demos/3dgs/?{urllib.parse.urlencode(query)}"
                                elif demo_params["type"] == "mesh":
                                    scene_data["mesh_demo_link"] = scene_data["demo_link"] = f"./demos/mesh/?{urllib.parse.urlencode(query)}"

                            # Add mesh demo
                            if os.path.exists(os.path.join(tmpdir, f"{method['id']}/{dataset['id']}/{scene}_mesh/params.json")):
                                with open(os.path.join(tmpdir, f"{method['id']}/{dataset['id']}/{scene}_mesh/params.json"), "r", encoding="utf8") as f:
                                    demo_params = json.load(f)
                                base = f"https://{SUPPLEMENTARY_RESULTS_REPOSITORY}/resolve/main/{method['id']}/{dataset['id']}/{scene}_mesh/"
                                query = {
                                    "p": base + "params.json",
                                }
                                if "links" in demo_params:
                                    for i, (label, link) in enumerate(demo_params["links"].items()):
                                        # Make link absolute
                                        if not link.startswith("http"):
                                            link = base + link
                                        # Url encode components
                                        query[f"p{i}"] = label
                                        query[f"p{i}v"] = link
                                scene_data["mesh_demo_link"] = f"./demos/mesh/?{urllib.parse.urlencode(query)}"

        configuration = {
            "include_docs": include_docs,
        }
        if include_docs is not None:
            root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            assert os.path.exists(os.path.join(root_path, "docs")), "docs directory not found"
            assert os.path.exists(os.path.join(root_path, ".git")), ".git directory not found"

            # Furthermore, we copy the latest docs dir to the temp directory.
            root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            configuration["docs_source_repo"] = root_path
        yield raw_data, configuration
