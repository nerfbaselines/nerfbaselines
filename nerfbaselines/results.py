import math
from typing import List, Dict, Any, cast, Union
import base64
import os
import struct
from pathlib import Path
import json
import warnings
import numpy as np
from . import metrics
from . import datasets
from . import registry
from .types import Literal, Optional, TypedDict, DatasetSpecMetadata
from ._constants import WEBPAGE_URL


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


def get_dataset_info(dataset: str) -> DatasetInfo:
    """
    Get the dataset info from the dataset repository.

    Args:
        dataset: The dataset name (type).

    Returns:
        The dataset info.
    """
    from .registry import get_dataset_spec
    metrics_info_path = Path(metrics.__file__).with_suffix(".json")
    assert metrics_info_path.exists(), f"Metrics info file {metrics_info_path} does not exist"
    metrics_info = json.loads(metrics_info_path.read_text(encoding="utf8"))
    dataset_info = cast(Optional[DatasetSpecMetadata], get_dataset_spec(dataset).get("metadata", None))
    if dataset_info is None:
        raise RuntimeError(f"Dataset {dataset} does not have metadata")

    # Fill metrics into dataset info
    metrics_dict = {v["id"]: v for v in metrics_info}
    return cast(DatasetInfo, {
        **dataset_info,
        "metrics": [metrics_dict[v] for v in dataset_info.get("metrics", [])],
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


def compile_dataset_results(results_path: Union[Path, str], dataset: str, scenes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compile the results.json file from the results repository.
    """
    results_path = Path(results_path)
    from . import registry

    try:
        dataset_info = cast(Dict[str, Any], get_dataset_info(dataset))
    except RuntimeError as e:
        if "does not have metadata" in str(e):
            dataset_info = {}
        else:
            raise e
    dataset_info_scenes = dataset_info.get("scenes", None)
    if scenes is not None:
        dataset_info_scenes = dataset_info["scenes"] = [(dataset_info_scenes or {}).get(x, dict(id=x)) for x in scenes]
    dataset_info["methods"] = []
    for method_id in os.listdir(results_path):
        # Skip methods not evaluated on the dataset
        if not any(results_path.joinpath(method_id, dataset).glob("*.json")):
            continue

        method_spec = registry.get_method_spec(method_id)
        method_data = method_spec.get("metadata", {}).copy()
        method_data["id"] = method_id
        method_data["scenes"] = {}

        # Load the results
        agg_metrics = {}

        local_scenes = dataset_info_scenes
        if local_scenes is None:
            local_scenes = [dict(id=x.stem) for x in results_path.joinpath(method_id, dataset).glob("*.json")]

        for scene in local_scenes:
            scene_id = scene["id"]
            scene_results_path = results_path.joinpath(method_id, dataset, scene_id + ".json")
            if not scene_results_path.exists():
                warnings.warn(f"Results file {scene_results_path} does not exist")
                continue
            scene_results = json.loads(scene_results_path.read_text(encoding="utf8"))
            results = load_metrics_from_results(scene_results)
            method_data["scenes"][scene_id] = {}
            for k, v in results.items():
                method_data["scenes"][scene_id][k] = mv = round(np.mean(v), 5)
                if k not in agg_metrics:
                    agg_metrics[k] = []
                agg_metrics[k].append(mv)

        if dataset_info_scenes is not None:
            for k, v in agg_metrics.items():
                if len(v) == len(dataset_info["scenes"]):
                    method_data[k] = round(np.mean(v), 5)
        dataset_info["methods"].append(method_data)
    return dataset_info


def get_benchmark_datasets() -> List[str]:
    """
    Get the list of registered benchmark datasets.
    """
    return [name for name in registry.get_supported_datasets() if registry.get_dataset_spec(name).get("metadata") is not None] 


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


def render_markdown_dataset_results_table(results, method_links: MethodLink = "none") -> str:
    """
    Generates a markdown table from the output of the `compile_dataset_results` method.

    Args:
        results: Output of the `nerfbaselines.results.compile_dataset_results` method.
    """
    columns = ["name"]
    table = [["Method"]]
    align = "l"
    column_orders: List[Optional[bool]] = [None]

    default_metric = results["default_metric"]
    for metric in results["metrics"]:
        columns.append(metric["id"])
        table[-1].append(metric["name"])
        column_orders.append(metric["ascending"])
        align += "r"

    # Add train time and GPU memory
    columns.append("total_train_time")
    table[-1].append("Time")
    column_orders.append(False)
    columns.append("gpu_memory")
    column_orders.append(False)
    table[-1].append("GPU mem.")
    align += "rr"

    def get(data, path):
        parts = path.split(".")
        for p in parts[:-1]:
            data = data.get(p, {})
        return data.get(parts[-1], None)

    # Add method's data
    ord_values = [[] for _ in column_orders]
    method_names = []
    default_metric_id = None
    for method in results["methods"]:
        method_names.append(method["name"])
        table.append([])
        for i, (column, asc) in enumerate(zip(columns, column_orders)):
            value = sort_value = get(method, column)
            if column == default_metric:
                default_metric_id = i
            if column == "name":
                # Render link if requested
                if method_links == "paper" and "paper_link" in method:
                    value = f"[{value}]({method['paper_link']})"
                elif method_links == "website" and "link" in method:
                    value = f"[{value}]({method['link']})"
                elif method_links == "results":
                    value = f"[{value}]({WEBPAGE_URL}/m-{method['id'].replace(':', '--')})"
            elif column == "total_train_time":
                value = format_duration(value)
            elif column == "gpu_memory":
                value = format_memory(value)
            elif isinstance(value, float):
                value = f"{value:.3f}"
            elif isinstance(value, int):
                value = f"{value:d}"
            elif value is None:
                value = "-"
            table[-1].append(value)
            if asc is None:
                continue
            if sort_value is not None:
                sort_value = -sort_value if asc else sort_value
            else:
                sort_value = float("inf")
            ord_values[i].append(sort_value)

    # Extract 1st,2nd,3rd places
    order_table = []
    for i, (asc, ord_vals) in enumerate(zip(column_orders, ord_values)):
        order_table.append(None)
        if asc is None:
            continue
        vals = sorted(list(set(ord_vals)))
        order_table[-1] = [vals.index(x) if x != float("inf") else None for x in ord_vals]

    def pad(value, align, cell_len):
        cell_len += 2
        value = f" {value} "
        padding = (cell_len - len(value)) * " "
        return (value + padding) if align == "l" else (padding + value)

    def pad_splitter(align, cell_len):
        cell_len += 2
        value = (cell_len - 1) * "-"
        return f":{value}" if align == "l" else f"{value}:"

    # Add bold to best numbers in column and italics to second values
    for i, order in enumerate(column_orders):
        if order is None:
            continue
        for j, val in enumerate(table[1:]):
            val = val[i]
            if order_table[i][j] == 0:
                val = f"**{val}**"
            elif order_table[i][j] == 1:
                val = f"*{val}*"
            table[j + 1][i] = val

    # Sort by the default metric
    all_metrics = ",".join(x["id"] for x in results["metrics"])
    if method_names:
        assert default_metric_id is not None, f"Default metric {default_metric} was not found in the set of metrics {all_metrics}."
        order = [x[-1] for x in sorted([(v, method_names[i], i) for i, v in enumerate(ord_values[default_metric_id])])]
        table = table[:1] + [table[i + 1] for i in order]

    cell_lens = [max(len(x[i]) for x in table) for i in range(len(table[0]))]
    table_str = ""
    table_str += "|" + "|".join(map(pad, table[0], align, cell_lens)) + "|\n"
    table_str += "|" + "|".join(map(pad_splitter, align, cell_lens)) + "|\n"
    for row in table[1:]:
        table_str += "|" + "|".join(map(pad, row, align, cell_lens)) + "|\n"
    return table_str
