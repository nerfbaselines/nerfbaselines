import logging
import subprocess
import copy
import click
import tempfile
from functools import partial
import math
import json
import os


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
    output_artifacts = method.get("output_artifacts", {})
    if "{dataset}/{scene}" in output_artifacts:
        output_artifact = output_artifacts["{dataset}/{scene}"]
        if output_artifact.get("link", None) is not None:
            return output_artifact["link"]

    resolved_paths = data.get("resolved_paths", {})
    link = resolved_paths.get(f"{method['id']}/{dataset}/{scene}.zip", None)
    return link


def get_dataset_data(raw_data):
    data = copy.deepcopy(raw_data)
    dataset = data["id"]
    default_metric = data.get("default_metric") or data["metrics"][0]["id"]
    sign = next((1 if m.get("ascending") else -1 for m in data["metrics"] if m["id"] == default_metric), None)
    if sign is None:
        sign = 1
    scenes_map = {s["id"]: s for s in data["scenes"]}
    data["slug"] = _clean_slug(raw_data["id"])
    data["methods"].sort(key=lambda x: sign * x.get(default_metric, -float("inf")))
    extended_metrics = data["metrics"] + [{"id": "total_train_time"}, {"id": "gpu_memory"}]
    data["methods"] = [{
        **m,
        "slug": _clean_slug(m["id"]),
        "average": {
            mid["id"]: _format_cell(_get_average([m["scenes"].get(s["id"]) or {} for s in data["scenes"]], mid["id"], sign), mid["id"])
            for mid in extended_metrics
        },
        "scenes": [{
            **{m["id"]: "-" for m in extended_metrics},
            **{k: _format_cell(v, k) for k, v in m["scenes"].get(s["id"], {}).items()},
            **scenes_map.get(s["id"], {}),
            "demo_link": None,
            "data_link": (
                _resolve_data_link(data, m, dataset, s["id"]) if s["id"] in m["scenes"]
                else None
            )
        } for s in data["scenes"]]
    } for m in data["methods"]]
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


def _generate_pages(data, input_path, output, base_path=""):
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    base_path = base_path.strip("/")
    if base_path:
        base_path = f"/{base_path}"
    env = Environment(
        loader=FileSystemLoader(os.path.join(input_path, "templates")),
        autoescape=select_autoescape()
    )
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



def build(input_path, output, raw_data, base_path=""):
    if os.path.exists(output):
        raise FileExistsError(f"Output directory {output} already exists.")

    # Generate all routes
    logging.info("Generating pages")
    data = get_data(raw_data)
    _generate_pages(data, input_path, output, base_path=base_path)

    # Copy static files
    logging.info("Copying static files")
    _copy_static_files(input_path, output)


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


def start_dev_server(raw_data):
    from livereload import Server  # type: ignore

    with tempfile.TemporaryDirectory() as output:
        input_path = os.path.dirname(os.path.abspath(__file__))

        # Build first version
        os.rmdir(output)
        build(input_path, output, raw_data)
        data = get_data(raw_data)

        def _on_dataloading_change():
            nonlocal data
            _reload_data_loading()
            new_data = get_data(raw_data)
            if json.dumps(data) != json.dumps(new_data):
                data = new_data
                _generate_pages(data, input_path, output)
                logging.info("Data reloaded")

        # Create server and watch for changes
        server = Server()
        SFH = server.SFH
        class HtmlRewriteSFHserver(SFH):
            def get(self, path, *args, **kwargs):
                fname = path.split("/")[-1]
                if fname and "." not in fname:
                    path = f"{path}.html"
                return super().get(path, *args, **kwargs)
        server.SFH = HtmlRewriteSFHserver
        logging.getLogger("tornado").setLevel(logging.WARNING)
        server.watch(os.path.join(input_path, "templates/**/*.html"), lambda: _generate_pages(data, input_path, output))
        server.watch(os.path.join(input_path, "public/**/*"), partial(_copy_static_files, input_path, output))
        server.watch(__file__, _on_dataloading_change)
        server._setup_logging = lambda: None
        logging.info("Starting dev server")
        server.serve(root=output)


def _get_method_licenses():
    from nerfbaselines.registry import get_supported_methods, get_method_spec

    implemented_methods = []
    for method in get_supported_methods():
        spec = get_method_spec(method).get("metadata", {})
        if ":" in method:
            continue
        if spec.get("licenses"):
            implemented_methods.append({"name": spec.get("name", method), "licenses": spec["licenses"]})
    implemented_methods.sort(key=lambda x: x.get("name", None))
    return implemented_methods


def _prepare_data(data_path, datasets=None):
    if data_path is not None:
        logging.info(f"Loading data from {data_path}")
        raw_data = get_raw_data(data_path)
        if datasets is None:
            from nerfbaselines.results import DEFAULT_DATASET_ORDER
            raw_data.sort(key=lambda x: DEFAULT_DATASET_ORDER.index(x["id"]) 
                          if x in DEFAULT_DATASET_ORDER 
                          else len(DEFAULT_DATASET_ORDER))
        if datasets is not None:
            raw_data_map = {d["id"]: d for d in raw_data}
            raw_data = [raw_data_map[d] for d in datasets]
    else:
        logging.info("Loading data from NerfBaselines repository")

        # Load data for all datasets
        from nerfbaselines.registry import get_dataset_downloaders
        from nerfbaselines.results import compile_dataset_results, DEFAULT_DATASET_ORDER

        if datasets is None:
            datasets = list(get_dataset_downloaders().keys())
            datasets.sort(key=lambda x: DEFAULT_DATASET_ORDER.index(x) 
                          if x in DEFAULT_DATASET_ORDER 
                          else len(DEFAULT_DATASET_ORDER))
            logging.info("Selected datasets: " + ", ".join(datasets))

        raw_data = []

        # Clone results repository
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.check_call("git clone https://huggingface.co/jkulhanek/nerfbaselines".split() + [tmpdir], env={"GIT_LFS_SKIP_SMUDGE": "1"})
            # List all paths in tmpdir
            existing_paths = [os.path.relpath(os.path.join(root, file), tmpdir) for root, _, files in os.walk(tmpdir) for file in files]
            resolved_paths = {
                path: f"https://huggingface.co/jkulhanek/nerfbaselines/resolve/main/{path}"
                for path in existing_paths
            }
            for dataset in datasets:
                dataset_info = compile_dataset_results(tmpdir, dataset)
                dataset_info["id"] = dataset
                dataset_info["resolved_paths"] = resolved_paths
                raw_data.append(dataset_info)
    return raw_data


def get_click_group():
    main = click.Group("web")

    @main.command("dev")
    @click.option("--data", "data_path", default=None, help="Path to data directory. If not provided, data is generated from the NerfBaselines repository.")
    @click.option("--datasets", default=None, help="List of comma separated dataset ids to include.")
    def _(data_path, datasets):
        from nerfbaselines.utils import setup_logging
        setup_logging(False)
        raw_data = _prepare_data(data_path, datasets.split(",") if datasets else None)
        start_dev_server(raw_data)

    @main.command("build")
    @click.option("--data", "data_path", default=None, help="Path to data directory. If not provided, data is generated from the NerfBaselines repository.")
    @click.option("--output", required=True, help="Output directory.")
    @click.option("--datasets", default=None, help="List of comma separated dataset ids to include.")
    @click.option("--base-path", default="", help="Base path for the website.")
    def _(output, data_path, datasets, base_path):
        from nerfbaselines.utils import setup_logging
        setup_logging(False)
        input_path = os.path.dirname(os.path.abspath(__file__))
        raw_data = _prepare_data(data_path, datasets.split(",") if datasets else None)
        build(input_path, output, raw_data, base_path=base_path)

    return main


if __name__ == "__main__":
    get_click_group()() 
