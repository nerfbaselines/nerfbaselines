import itertools
import numpy as np
import pprint
import shutil
import logging
import pprint
import json
import warnings
import tempfile
import os
import click
from nerfbaselines.utils import setup_logging, handle_cli_error
from nerfbaselines.utils import run_inside_eval_container
from nerfbaselines.datasets import load_dataset
from nerfbaselines.io import open_any_directory, deserialize_nb_info, serialize_nb_info
from nerfbaselines.evaluation import evaluate
from nerfbaselines.registry import resolve_evaluation_protocol
from nerfbaselines.io import save_output_artifact


def build_changes_tracker():
    _changes = {}
    _has_changes = False

    def _add_changes(path, obj1, obj2=None, indent=2, only_diff=False):
        nonlocal _has_changes
        def _changes_append(path, value):
            changes = _changes
            for p in path:
                if p not in changes:
                    changes[p] = {}
                changes = changes[p]
            changes[None] = changes.get(None, [])
            changes[None].append(value)
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            out = False
            o1keys = set(obj1.keys())
            for k in itertools.chain(obj1.keys(), (k for k in obj2.keys() if k not in o1keys)):
                if k not in obj1:
                    v = pprint.pformat(obj2[k])[:20]
                    _changes_append(path, f"\033[32m{k}: {v}\033[0m")
                    _has_changes = True
                    out = True
                    continue
                if k not in obj2:
                    v = pprint.pformat(obj1[k])[:20]
                    _changes_append(path, f"\033[9;31m{k}: {v}\033[0m")
                    _has_changes = True
                    out = True
                    continue
                if _add_changes(path + [k], obj1.get(k, None), obj2.get(k, None), indent=indent, only_diff=only_diff):
                    out = True
            return out
        v1, v2 = pprint.pformat(obj1), pprint.pformat(obj2)
        if v1 != v2:
            _changes_append(path[:-1], f"{path[-1]}: \033[9;31m{v1[:10]}\033[32m{v2[:10]}\033[0m")
            if "datetime" in path[-1].lower() or "version" in path[-1].lower():
                return False
            _has_changes = True
            return True
        else:
            if not only_diff:
                _changes_append(path[:-1], f"{path[-1]}: {v1[:20]}")
            return False

    def _print_changes(indent=2, _values=None, _offset=0):
        if _values is None:
            _values = _changes
        if None in _values:
            for v in _values[None]:
                print(" "*indent*_offset + v)
        for k, v in _values.items():
            if k is None:
                continue
            print(" "*indent*_offset + k + ":")
            _print_changes(indent, v, _offset+1)
    def _has_path(path):
        changes = _changes
        for p in path:
            if p not in changes:
                return False
            changes = changes[p]
        return True
    return _add_changes, _print_changes, lambda: _has_changes, _has_path


def build_dir_tree(path):
    if os.path.isfile(path):
        return os.path.basename(path)
    if os.path.isdir(path):
        return {os.path.basename(f): build_dir_tree(os.path.join(path, f)) for f in os.listdir(path)}
    raise RuntimeError(f"Path {path} is not a file or directory")


@click.command("fix-output-artifact")
@click.option("--input", type=str, default=None, required=True)
@click.option("--data", type=str, default=None, required=False)
@click.option("--method", "method_name", type=str, default=None, required=False)
@click.option("--rerun-evaluation", is_flag=True)
@click.option("--output", "new_artifact", type=str, required=False, help="Path to save the new output artifact")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--force", is_flag=True)
@click.option("--inplace", is_flag=True)
@handle_cli_error
def main(input: str,
         data=None,
         new_artifact=None,
         inplace: bool = False,
         method_name=None,
         rerun_evaluation: bool = False,
         force: bool = False,
         verbose: bool = False):
    setup_logging(verbose)
    if not inplace and new_artifact is None:
        raise RuntimeError("Please specify --new-artifact or --inplace to overwrite the input artifact")
    if inplace:
        new_artifact = input
    else:
        assert new_artifact is not None, "Please specify --new-artifact to save the new artifact"
        if os.path.exists(new_artifact):
            raise RuntimeError(f"New artifact path {new_artifact} already exists")

    errors, skips, successes = [], [], []
    _add_changes, _print_changes, _has_changes, _has_path = build_changes_tracker()

    def mark_success(message: str):
        logging.info(message)
        successes.append(message)

    def mark_error(message: str):
        logging.error(message)
        errors.append(message)

    def mark_skip(message: str):
        logging.warning(message)
        skips.append(message)

    basename = "artifact.zip"
    basename = os.path.split(new_artifact)[-1]
    with open_any_directory(input, "r") as inpath, \
            tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "output")
        os.makedirs(outpath, exist_ok=True)
        assert os.path.exists(inpath), f"Input path {input} does not exist"
        assert os.path.isdir(inpath), f"Input path {input} is not a directory"
        assert os.path.exists(os.path.join(inpath, "results.json")), f"Input path {input}/results.json does not exist"

        with open(os.path.join(inpath, "results.json"), "r") as f:
            results_data = json.load(f)

        print("Results: ", pprint.pformat(results_data))

        # If data is not provided, try to load it as external dataset
        _data = data
        info = results_data.get("nb_info", results_data.get("info", {}))
        print("Info: ", pprint.pformat(info))
        dm = info.get("dataset_metadata") or {}
        scene = dm.get("scene", info.get("dataset_scene"))
        dataset = dm.get("name", info.get("dataset_name", info.get("dataset_type")))
        new_data = f"external://{dataset}/{scene}"
        if data is None:
            data = new_data
            warnings.warn(f"Using external dataset {data}, if this is not correct, please specify --data argument")
        else:
            if data != new_data:
                warnings.warn(f"Using external dataset {data}, but results data indicates {new_data}")
                if force:
                    warnings.warn("Forcing to use the provided dataset")
                else:
                    raise RuntimeError("Dataset mismatch, use --force to override")
        evaluation_protocol = info.get("evaluation_protocol", 
                                       dm.get("evaluation_protocol", "default"))

        # Verify that the dataset exists
        test_dataset = None
        if data.startswith("external://") or _data is not None:
            logging.info(f"Loading external dataset {data}")
            train_dataset = load_dataset(data, split="train", load_features=False)
            test_dataset = load_dataset(data, split="test", load_features=True)
            mark_success(f"Loaded external dataset {data}")

            # Validate evaluation protocol
            dataset_evaluation_protocol = train_dataset["metadata"].get("evaluation_protocol", "default")
            if evaluation_protocol != dataset_evaluation_protocol:
                mark_error(f"Dataset evaluation protocol {dataset_evaluation_protocol} does not match the evaluation protocol {evaluation_protocol} in the results data")
                if force:
                    warnings.warn("Forcing to use the evaluation protocol from the dataset {dataset_evaluation_protocol}")
                    evaluation_protocol = dataset_evaluation_protocol
                else:
                    raise RuntimeError("Evaluation protocol mismatch, use --force to override")
            else:
                mark_success(f"Dataset evaluation protocol {dataset_evaluation_protocol} matches the evaluation protocol in the results data")
        else:
            mark_skip(f"Skipping dataset loading for {data}. Please provide --data argument to load the dataset")
            mark_skip("Skipping correct evaluation protocol validation")

        # Validate checkpoint if requested
        validate_checkpoint = False
        checkpoint_path = os.path.join(inpath, "checkpoint")
        tensorboard_path = os.path.join(inpath, "tensorboard")
        if validate_checkpoint:
            mark_success("Checkpoint validation passed")
        else:
            mark_skip("Skipping checkpoint validation")

        # Re-run render if requested
        # TODO: implement this
        rerun_render = False
        predictions_path = os.path.join(outpath, "predictions")
        if rerun_render:
            mark_success("Rendered predictions matched the ones stored in the artifact")
        else:
            shutil.copytree(os.path.join(inpath, "predictions"), os.path.join(outpath, "predictions"))
            if test_dataset is not None:
                with open(os.path.join(outpath, "predictions", "info.json"), "r") as f:
                    oldtext = f.read()
                    old_info = json.loads(oldtext)
                new_info = deserialize_nb_info(old_info.copy())

                # Clear legacy fields
                new_info.pop("dataset_scene", None)
                new_info.pop("dataset_type", None)
                new_info.pop("dataset_name", None)
                new_info.pop("dataset_background_color", None)
                new_info.pop("expected_scene_scale", None)

                new_info["dataset_metadata"] = test_dataset["metadata"].copy()
                background_color = test_dataset["metadata"].get("background_color", None)
                if isinstance(background_color, np.ndarray):
                    new_info["dataset_metadata"]["background_color"] = background_color.tolist()
                new_info = serialize_nb_info(new_info)
                pprint.pprint(new_info)
                with open(os.path.join(outpath, "predictions", "info.json"), "w") as f:
                    json.dump(new_info, f, indent=2)
                with open(os.path.join(outpath, "predictions", "info.json"), "r") as f:
                    newtext = f.read()
                if not _add_changes(["predictions", "info.json"], old_info, new_info):
                    _add_changes(["predictions", "info.json"], oldtext, newtext)

                # TODO: compare GT images 
            mark_skip("Skipping rendering predictions. Please use --rerun-render to re-run rendering")

        # Re-run evaluation if requested
        metrics_path = os.path.join(inpath, "results.json")
        if rerun_evaluation or rerun_render:
            metrics_path = os.path.join(outpath, "results.json")
            logging.info(f"Re-running evaluation using evaluation protocol: {evaluation_protocol}")
            with run_inside_eval_container():
                evaluation_protocol_instance = resolve_evaluation_protocol(evaluation_protocol)
                evaluate(os.path.join(outpath, "predictions"), os.path.join(outpath, "results.json"), evaluation_protocol=evaluation_protocol_instance)

            # Now, we compare the predictions
            if _add_changes(["results.json"], results_data, json.load(open(os.path.join(outpath, "results.json")))):
                mark_error("New evaluated results did not match the ones stored in the artifact")
            else:
                mark_success("New evaluated results matched the ones stored in the artifact")
        else:
            # Copy results.json
            mark_skip("Skipping evaluation. Please use --rerun-evaluation to re-run evaluation, or --rerun-render to re-run rendering")

        # Saving output artifact
        save_output_artifact(
            checkpoint_path,
            predictions_path,
            metrics_path,
            tensorboard_path,
            os.path.join(tmpdir, basename),
            validate=False)

        # Track missing files
        filetree1 = build_dir_tree(inpath)
        with open_any_directory(os.path.join(tmpdir, basename), "r") as _outpath:
            filetree2 = build_dir_tree(_outpath)
        _add_changes([], filetree1, filetree2, only_diff=True)

        print()
        print("Changes:")
        _print_changes()

        shutil.copyfile(os.path.join(tmpdir, basename), new_artifact)
        logging.info(f"New artifact is stored at {new_artifact}")
