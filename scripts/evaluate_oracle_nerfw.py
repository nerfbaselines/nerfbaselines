import os
import argparse
import numpy as np
import logging
from typing import Dict, Union, Iterable
from nerfbaselines import EvaluationProtocol, Dataset, Method, RenderOutput, RenderOptions
from nerfbaselines.evaluation import compute_metrics, render_all_images
from nerfbaselines.io import save_evaluation_results, get_predictions_sha
from nerfbaselines.utils import image_to_srgb
from nerfbaselines.datasets import load_dataset, dataset_index_select
from nerfbaselines import load_checkpoint


class NerfWOracleEvaluationProtocol(EvaluationProtocol):
    def __init__(self):
        self._compute_metrics = compute_metrics

    def get_name(self):
        return "nerfw-oracle"

    def render(self, method: Method, dataset: Dataset, *, options=None) -> RenderOutput:
        dataset["cameras"].item()  # Assert single camera
        embedding = (options or {}).get("embedding", None)
        optim_result = None
        try:
            optim_result = method.optimize_embedding(dataset, embedding=embedding)
            embedding = optim_result["embedding"]
        except NotImplementedError as e:
            logging.debug(e)
            method_id = method.get_method_info()["method_id"]
            logging.warning(f"Method {method_id} does not support camera embedding optimization.")

        new_options: RenderOptions = {
            **(options or {}),
            "embedding": embedding,
        }
        return method.render(dataset["cameras"], options=new_options)

    def evaluate(self, predictions: RenderOutput, dataset: Dataset) -> Dict[str, Union[float, int]]:
        assert len(dataset["images"]) == 1, "EvaluationProtocol.evaluate must be run on individual samples (a dataset with a single image)"
        gt = dataset["images"][0]
        color = predictions["color"]

        background_color = dataset["metadata"].get("background_color", None)
        color_srgb = image_to_srgb(color, np.uint8, color_space="srgb", background_color=background_color)
        gt_srgb = image_to_srgb(gt, np.uint8, color_space="srgb", background_color=background_color)
        metrics = self._compute_metrics(color_srgb, gt_srgb)
        return metrics

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        acc = {}
        for i, data in enumerate(metrics):
            for k, v in data.items():
                acc[k] = (acc.get(k, 0) * i + v) / (i + 1)
        return acc


if __name__ == "__main__":
    # Render with the oracle evaluation protocol
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    dataset = load_dataset(args.data, "test")

    # Sort the dataset by name
    indices = np.argsort(dataset["image_paths"])
    dataset = dataset_index_select(dataset, indices)

    predictions_path = os.path.join(args.output, "predictions")

    # Render and evaluate with the oracle evaluation protocol
    evaluation_protocol = NerfWOracleEvaluationProtocol()
    metrics_list = []
    metrics_lists = {}
    with load_checkpoint(args.checkpoint) as (method, nb_info):
        for i, pred in enumerate(render_all_images(method, 
                                                   dataset, 
                                                   predictions_path, 
                                                   nb_info=nb_info, evaluation_protocol=evaluation_protocol)):
            dataset_slice = dataset_index_select(dataset, [i])
            metrics = evaluation_protocol.evaluate(pred, dataset_slice)
            metrics_list.append(metrics)
            for k, v in metrics.items():
                if k not in metrics_lists:
                    metrics_lists[k] = []
                metrics_lists[k].append(v)

    metrics = evaluation_protocol.accumulate_metrics(metrics_list)
    predictions_sha, ground_truth_sha = get_predictions_sha(str(predictions_path))

    # If output is specified, write the results to a file
    out = save_evaluation_results(os.path.join(args.output, "results.json"),
                                  metrics=metrics, 
                                  metrics_lists=metrics_lists, 
                                  predictions_sha=predictions_sha,
                                  ground_truth_sha=ground_truth_sha,
                                  evaluation_protocol=evaluation_protocol.get_name(),
                                  nb_info=nb_info)
