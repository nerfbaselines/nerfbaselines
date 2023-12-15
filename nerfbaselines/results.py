from typing import List, Dict
import base64
import os
import struct
from pathlib import Path
import json
import warnings
import numpy as np
from . import metrics
from . import datasets


def get_dataset_info(dataset: str) -> Dict:
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
    dataset_info = json.loads(Path(datasets.__file__).absolute().parent.joinpath(dataset + ".json").read_text(encoding="utf8"))

    # Fill metrics into dataset info
    metrics_dict = {v["id"]: v for v in metrics_info}
    dataset_info["metrics"] = [metrics_dict[v] for v in dataset_info["metrics"]]
    return dataset_info


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
    return out


def compile_dataset_results(results_path: Path, dataset: str) -> Dict:
    """
    Compile the results.json file from the results repository.
    """
    from . import registry

    dataset_info = get_dataset_info(dataset)
    dataset_info["methods"] = []
    for method_id in os.listdir(results_path):
        method_spec = registry.get(method_id)
        method_data = method_spec.metadata
        method_data["id"] = method_id
        method_data["scenes"] = {}

        # Load the results
        agg_metrics = {}
        for scene in dataset_info["scenes"]:
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

        for k, v in agg_metrics.items():
            if len(v) == len(dataset_info["scenes"]):
                method_data[k] = round(np.mean(v), 5)
        dataset_info["methods"].append(method_data)
    return dataset_info


def get_benchmark_datasets() -> List[str]:
    """
    Get the list of registered benchmark datasets.
    """
    return [x.with_suffix("").name for x in (Path(datasets.__file__).absolute().parent.glob("*.json"))]
