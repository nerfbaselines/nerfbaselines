import traceback
import sys
import json
import glob
import pprint
import logging
import os
import numpy as np
import click
from PIL import Image
import tempfile
from tqdm import trange
from nerfbaselines import backends
from nerfbaselines import registry
from nerfbaselines.datasets import load_dataset, dataset_index_select
from nerfbaselines.utils import SetParamOptionType, handle_cli_error, setup_logging, Indices
from nerfbaselines.utils import run_inside_eval_container
from nerfbaselines.logging import TensorboardLogger
from nerfbaselines.io import open_any_directory
from nerfbaselines.training import Trainer, eval_few, eval_all
from nerfbaselines.registry import build_evaluation_protocol
from nerfbaselines.evaluation import evaluate
from ._common import ChangesTracker


@click.command("test-method")
@click.option("--method", "method_name", type=click.Choice(sorted(registry.get_supported_methods())), required=True, help="Method to use")
@click.option("--data", "dataset", type=str, required=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--set", "config_overrides", help="Override a parameter in the method.", type=SetParamOptionType(), multiple=True, default=None)
@click.option("--fast", is_flag=True, help="Run only the fast tests")
@click.option("--steps", type=int, default=2013, help="Number of steps to run")
@handle_cli_error
def main(method_name: str, 
         dataset: str, *, 
         backend_name=None, 
         verbose: bool = False,
         config_overrides=None,
         steps: int = 113,
         fast=False):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    # For some methods we do less steps by default to make it faster
    parameter_source = click.get_current_context().get_parameter_source('steps')
    if parameter_source == click.core.ParameterSource.DEFAULT:
        if method_name in ("nerf", "mipnerf", "mipnerf360"):
            steps = 113

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
    method_spec = registry.get_method_spec(method_name)

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
            if fast:
                train_dataset = dataset_index_select(train_dataset, list(range(min(len(train_dataset["cameras"]), 10))))
            logging.info("Train dataset: \n" + pprint.pformat(train_dataset["metadata"]))
            assert train_dataset["cameras"].image_sizes is not None, "image sizes must be specified"
            mark_success("Train dataset loaded")

            # Apply config overrides for the train dataset
            dataset_name = train_dataset["metadata"].get("name")
            if dataset_name is not None:
                _dataset_overrides = (method_spec.get("dataset_overrides") or {}).get(dataset_name, {}) or {}
                for k, v in _dataset_overrides.items():
                    config_overrides = config_overrides or {}
                    if k not in config_overrides:
                        config_overrides[k] = v
            logging.info("Config overrides: \n" + pprint.pformat(config_overrides))

            # Load eval dataset
            test_dataset = load_dataset(dataset, 
                                        split="test", 
                                        features=method_info.get("required_features"), 
                                        supported_camera_models=method_info.get("supported_camera_models"), 
                                        load_features=True)
            if fast:
                test_dataset = dataset_index_select(test_dataset, list(range(min(len(test_dataset["cameras"]), 10))))
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
            for i in trange(steps, desc="Training"):
                model.train_iteration(i)
            mark_success("Train iteration passes")

            with tempfile.TemporaryDirectory() as tmpdir_logger:
                # Test eval_few
                logger = TensorboardLogger(tmpdir_logger)
                eval_protocol = build_evaluation_protocol(test_dataset["metadata"]["evaluation_protocol"])
                eval_few(model, logger, test_dataset, split="test", step=steps, evaluation_protocol=eval_protocol)
                mark_success("Eval few passes")

                # Test eval_all
                nb_info = {}
                eval_all(model, logger, dataset_index_select(test_dataset, [0]), 
                         step=steps, evaluation_protocol=eval_protocol,
                         split="test", nb_info=nb_info, output=tmpdir_logger)
                mark_success("Eval all passes")

            render_out = None
            for render_out in model.render(test_dataset["cameras"][:1]):
                pass
            assert render_out is not None, "Render output is None" 
            logging.info("Render output: " + pprint.pformat({
                k: getattr(v, "shape", None)
                for k, v in render_out.items()}))
            mark_success("Render works")

            with tempfile.TemporaryDirectory() as tmpdir:
                # Test running the evaluation
                model.save(os.path.join(tmpdir, "ckpt1"))
                close_method = getattr(model, "close", None)
                if close_method is not None:
                    close_method()
                del close_method
                del model
                mark_success("Saving works")

                # Load from checkpoint
                model2 = None
                try:
                    model2 = method_cls(checkpoint=os.path.join(tmpdir, "ckpt1"))
                    model2_info = model2.get_info()
                    print("Loaded model info: \n", pprint.pformat(model2_info))
                    assert model2_info.get("loaded_step", None) == steps, f"Loaded step is not correct {model2_info.get('loaded_step')} != {steps}"
                    mark_success("Loading from checkpoint (without train dataset) passes")
                    model2.save(os.path.join(tmpdir, "ckpt2"))
                    del model2_info
                except Exception:
                    traceback.print_exc()
                    mark_error("Loading from checkpoint (without train dataset) fails")

                # Compare the checkpoints
                if not os.path.exists(os.path.join(tmpdir, "ckpt2")):
                    mark_skip("Resaving method test skiped (no checkpoint)")
                else:
                    tracker = ChangesTracker()
                    if tracker.add_dir_changes((), os.path.join(tmpdir, "ckpt1"), os.path.join(tmpdir, "ckpt2")):
                        tracker.print_changes()
                        mark_error("Resaving method does not yield the same checkpoint")
                    else:
                        mark_success("Resaving method yields same checkpoint")

                def test_render(model2, post):
                    render2_out = None
                    for render2_out in model2.render(test_dataset["cameras"][:1]):
                        pass
                    assert render2_out is not None, "Render output is None"
                    logging.info("Render loaded model output: \n" + pprint.pformat({
                        k: getattr(v, "shape", v)
                        for k, v in render2_out.items()}))

                    # Compare the outputs
                    try:
                        assert len(render_out) == len(render2_out)
                        for k, v in render_out.items():
                            assert k in render2_out
                            v2 = render2_out[k]
                            assert getattr(v, "shape", v) == getattr(v2, "shape", v2)
                            assert isinstance(v, np.ndarray)
                            assert isinstance(v2, np.ndarray)
                            np.testing.assert_allclose(v, v2, atol=1e-6)
                        mark_success(f"Restored model {post} matches original")
                    except AssertionError:
                        traceback.print_exc()
                        mark_error(f"Restored model {post} does not match original")
                    del model2
                    del render2_out

                if model2 is not None:
                    test_render(model2, "(without train dataset)")
                else:
                    mark_skip("Skipping render comparison (without train dataset)")


                # Load from checkpoint (with train dataset)
                model2 = None
                try:
                    model2 = method_cls(
                        checkpoint=os.path.join(tmpdir, "ckpt1"),
                        train_dataset=train_dataset,
                        config_overrides=config_overrides,
                    )
                    model2_info = model2.get_info()
                    print("Loaded model info: \n", pprint.pformat(model2_info))
                    assert model2_info.get("loaded_step", None) == steps, f"Loaded step is not correct {model2_info.get('loaded_step')} != {steps}"
                    mark_success("Loading from checkpoint (with train dataset) passes")
                    del model2_info
                except Exception:
                    traceback.print_exc()
                    mark_error("Loading from checkpoint fails")

                if model2 is not None:
                    test_render(model2, "(with train dataset)")
                else:
                    mark_skip("Skipping render comparison (with train dataset)")

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
                save_iters=Indices([]),
                eval_all_iters=Indices([-1]),
                eval_few_iters=Indices([2]),
                loggers=frozenset(("tensorboard",)),
                generate_output_artifact=True,
                config_overrides=config_overrides,
            )
            if fast:
                trainer.num_iterations = max(10, steps)
                # Fix total for indices
                for v in vars(trainer).values():
                    if isinstance(v, Indices):
                        v.total = trainer.num_iterations + 1
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
                            if os.path.isdir(k):
                                continue
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
        if fast:
            mark_skip("Skipping paper results comparison for fast test")
        elif "paper_results" in metadata:
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
