import traceback
import sys
import json
import glob
import pprint
import logging
import os
import numpy as np
from .datasets import load_dataset, dataset_index_select
from .evaluate import run_inside_eval_container
from .utils import SetParamOptionType, handle_cli_error, setup_logging, Indices
from .logging import TensorboardLogger
from .io import open_any_directory
from . import registry
from . import backends
import click
from . import registry
from .train import Trainer, eval_few, eval_all
from .evaluate import evaluate, get_evaluation_protocol
from PIL import Image
import tempfile


@click.command("test-method")
@click.option("--method", "method_name", type=click.Choice(sorted(registry.get_supported_methods())), required=True, help="Method to use")
@click.option("--data", "dataset", type=str, required=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--set", "config_overrides", help="Override a parameter in the method.", type=SetParamOptionType(), multiple=True, default=None)
@handle_cli_error
def main(method_name: str, 
         dataset: str, *, 
         backend_name=None, 
         verbose: bool = False,
         config_overrides=None):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    if config_overrides is not None and isinstance(config_overrides, (list, tuple)):
        config_overrides = dict(config_overrides)

    errors, skips, successes = [], [], []

    def mark_success(message: str):
        logging.info(message)
        successes.append(message)

    def mark_error(message: str):
        logging.error(message)
        errors.append(message)

    def mark_skip(message: str):
        logging.warning(message)
        skips.append(message)

    # Get method spec
    method_spec = registry.get(method_name)

    # Load train dataset
    logging.info("loading train dataset")

    output_context = tempfile.TemporaryDirectory()
    with output_context as output:
        with registry.build_method(method_name, backend_name) as method_cls:
            mark_success("Method backend initialized")

            method_info = method_cls.get_method_info()
            logging.info("Method info: \n" + pprint.pformat(method_info))
            mark_success("Method info loaded")

            # Install
            method_cls.install()
            mark_success("Method installed")

            # Load train dataset
            train_dataset = load_dataset(dataset, 
                                         split="train", 
                                         features=method_info.get("required_features"), 
                                         supported_camera_models=method_info.get("supported_camera_models"), 
                                         load_features=True)
            logging.info("Train dataset: \n" + pprint.pformat(train_dataset["metadata"]))
            assert train_dataset["cameras"].image_sizes is not None, "image sizes must be specified"
            mark_success("Train dataset loaded")

            # Load eval dataset
            test_dataset = load_dataset(dataset, 
                                        split="test", 
                                        features=method_info.get("required_features"), 
                                        supported_camera_models=method_info.get("supported_camera_models"), 
                                        load_features=True)
            test_dataset["metadata"]["expected_scene_scale"] = train_dataset["metadata"].get("expected_scene_scale")
            mark_success("Test dataset loaded")

            # Build the method
            model = method_cls(
                checkpoint=None,
                train_dataset=train_dataset,
                config_overrides=config_overrides,
            )
            model_info = model.get_info()
            logging.info("Method info: " + pprint.pformat(model_info))
            mark_success("Model initialized")
            del model_info

            # Test running the training
            for i in range(13):
                model.train_iteration(i)
            mark_success("Train iteration passes")

            with tempfile.TemporaryDirectory() as tmpdir_logger:
                # Test eval_few
                logger = TensorboardLogger(tmpdir_logger)
                test_dataset_name = test_dataset['metadata'].get("name")
                eval_protocol = get_evaluation_protocol(dataset_name=test_dataset_name)
                eval_few(model, logger, test_dataset, split="test", step=13, evaluation_protocol=eval_protocol)
                mark_success("Eval few passes")

                # Test eval_all
                nb_info = {}
                eval_all(model, logger, dataset_index_select(test_dataset, [0]), 
                         step=13, evaluation_protocol=eval_protocol,
                         split="test", nb_info=nb_info, output=tmpdir_logger)
                mark_success("Eval all passes")


            for render_out in model.render(test_dataset["cameras"][:1]):
                pass
            logging.info("Render output: " + pprint.pformat({
                k: v.shape 
                for k, v in render_out.items()}))
            mark_success("Render works")

            with tempfile.TemporaryDirectory() as tmpdir:
                # Test running the evaluation
                model.save(tmpdir)
                if hasattr(model, "close"):
                    model.close()
                del model
                mark_success("Saving works")

                # Load from checkpoint
                model2 = None
                try:
                    model2 = method_cls(
                        checkpoint=tmpdir,
                        train_dataset=train_dataset,
                        config_overrides=config_overrides,
                    )
                    model2_info = model2.get_info()
                    print("Loaded model info: \n", pprint.pformat(model2_info))
                    assert model2_info["loaded_step"] == 13
                    mark_success("Loading from checkpoint passes")
                    del model2_info
                except Exception:
                    traceback.print_exc()
                    mark_error("Loading from checkpoint fails")

                if model2 is not None:
                    for render2_out in model2.render(test_dataset["cameras"][:1]):
                        pass
                    logging.info("Render loaded model output: \n" + pprint.pformat({
                        k: v.shape 
                        for k, v in render2_out.items()}))

                    # Compare the outputs
                    try:
                        assert len(render_out) == len(render2_out)
                        for k, v in render_out.items():
                            assert k in render2_out
                            assert v.shape == render2_out[k].shape
                            v2 = render2_out[k]
                            np.testing.assert_allclose(v, v2)
                        mark_success("Restored model matches original")
                    except AssertionError:
                        traceback.print_exc()
                        mark_error("Restored model does not match original")
                    del model2
                    del render_out
                    del render2_out

                else:
                    mark_skip("Skipping render comparison")

            model = method_cls(
                checkpoint=None,
                train_dataset=train_dataset,
                config_overrides=config_overrides,
            )

            trainer = Trainer(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                method=model,
                output=output,
                save_iters=(),
                eval_all_iters=Indices([-1]),
                eval_few_iters=Indices([2]),
                loggers=frozenset(("tensorboard",)),
                generate_output_artifact=True,
                config_overrides=config_overrides,
            )
            trainer.train()
            logging.info("Training finished")

            # Print the output metrics
            result_files = glob.glob(os.path.join(output, "results-*.json"))
            if not result_files:
                raise RuntimeError("results-*.json not found")
            predictions_files = glob.glob(os.path.join(output, "predictions-*.tar.gz"))
            if not result_files:
                raise RuntimeError("predictions-*.tar.gz not found")
            checkpoints_files = glob.glob(os.path.join(output, "checkpoint-*"))
            if not checkpoints_files:
                raise RuntimeError("checkpoint-* not found")
            results_filename = max(
                result_files,
                key=lambda x: int(x.split("-")[-1].split(".")[0]))
            predictions_filename = max(
                predictions_files,
                key=lambda x: int(x.split("-")[-1].split(".")[0]))
            checkpoint_filename = max(
                checkpoints_files,
                key=lambda x: int(x.split("-")[-1].split(".")[0]))
            with open(results_filename, "r") as f:
                results = json.load(f)
            logging.info("Results: \n" + pprint.pformat(results))
            del trainer
            mark_success("Full training works")
            del model

            # Test if the results can be reproduced from the checkpoint
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    model2 = method_cls(checkpoint=checkpoint_filename)
                    eval_all(model2, None, test_dataset, step=30, evaluation_protocol=eval_protocol, split="test", nb_info={}, output=tmpdir)
                    assert os.path.exists(os.path.join(tmpdir, "predictions-30.tar.gz"))

                    with open_any_directory(os.path.join(tmpdir, "predictions-30.tar.gz")) as preddir, open_any_directory(predictions_filename) as predrefdir:
                        for k in glob.glob(preddir + "/color/*"):
                            assert os.path.exists(os.path.join(predrefdir, "color", k.split("/")[-1]))
                            im1 = np.array(Image.open(k))
                            im2 = np.array(Image.open(os.path.join(predrefdir, "color", k.split("/")[-1])))
                            np.testing.assert_equal(im1, im2)
                    del model2
                mark_success("Checkpoint reproduces results")
            except Exception:
                traceback.print_exc()
                mark_error("Checkpoint does not reproduce results")

        metrics = results["metrics"]
        logging.info("Metrics: \n" + pprint.pformat(metrics))

        # Test if evaluation on the computed results matches the expected values
        with run_inside_eval_container(backend_name), tempfile.TemporaryDirectory() as tmpdir:
            evaluate(predictions_filename, os.path.join(tmpdir, "eval.json"))
            with open(os.path.join(tmpdir, "eval.json"), "r") as f:
                eval_results = json.load(f)
        logging.info("Eval results: \n" + pprint.pformat(eval_results))
        logging.info("Eval metrics: \n" + pprint.pformat(eval_results["metrics"]))
        # Compare the metrics
        assert len(metrics) == len(eval_results["metrics"])
        has_error = False
        for k, v in metrics.items():
            assert k in eval_results["metrics"]
            tol = 1e-6
            if "lpips" in k:
                tol = 5e-4
            if abs(v - eval_results["metrics"][k]) >= tol:
                logging.error(f"Metric {k} does not match: computed {v}, expected {eval_results['metrics'][k]}, tolerance {tol}")
                has_error = True
        if not has_error:
            mark_success("Final evaluation works and matches predictions")
        else:
            mark_error("Final evaluation does not match predictions")


        # Collect the metrics and compare with the expected values
        # Test evaluation command - if the results match the expected values 
        # with run_inside_eval_container(backend_name):
        #     evaluate(predictions, output)
        metadata = method_spec.get("metadata", {})
        if "paper_results" in metadata:
            method_key = train_dataset["metadata"]["name"] + "/" + train_dataset["metadata"]["scene"]
            paper_results = metadata["paper_results"].get(method_key, None)
            if paper_results is not None:
                logging.info("Paper results: \n" + pprint.pformat(paper_results))
                logging.info("Method results: \n" + pprint.pformat({
                    k: metrics[k]
                    for k in paper_results.keys()
                }))
                match = True
                for k, v in paper_results.items():
                    assert k in metrics
                    tolerance = 0.2
                    if k == "ssim":
                        tolerance = 0.05
                    if not abs(metrics[k] - v) < tolerance:
                        logging.error(f"{k} not within tolerance: computed {metrics[k]}, expected {v}, tolerance {tolerance}")
                        match = False
                if match:
                    mark_success("Paper results match")
                else:
                    mark_error(f"Paper results do not match for {method_key}")
            else:
                mark_skip(f"No paper results found for {method_key}")
        else:
            mark_skip(f"No paper results for method {method_name}")

        # TODO: If the method can create a demo, test generating the demo
        # TODO: Test running the viewer

        # TODO: print error summary
        print("Summary:")
        for message in successes:
            print(f"  \033[92m\u2713 {message}\033[0m")
        for message in skips:
            print(f"  \033[93m\u26A0 {message}\033[0m")
        for message in errors:
            print(f"  \033[91m\u2717 {message}\033[0m")
        sys.exit(1 if errors else 0)
