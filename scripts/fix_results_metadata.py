import os
import json
import argparse
from pathlib import Path
import tarfile
import tempfile
import shutil
from nerfbaselines.io import save_output_artifact


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to the dataset directory.")
parser.add_argument("--dry-run", action="store_true", help="If set, do not write changes to the files.")
args = parser.parse_args()


path = Path(args.path)
for method_path in path.iterdir():
    if not method_path.is_dir():
        continue
    for dataset_path in method_path.iterdir():
        dataset = dataset_path.name
        if not dataset_path.is_dir():
            continue
        for scene_path in dataset_path.iterdir():
            scene = scene_path.name
            results_path = None
            has_change = False
            def setval(val, key, value):
                global has_change
                assert key in val
                if val[key] != value:
                    val[key] = value
                    has_change = True
            with tempfile.TemporaryDirectory() as tmpdir:
                predictions_paths = scene_path.glob("predictions-*")
                for predictions_targz_path in scene_path.glob("predictions-*.tar.gz"):
                    predictions_path = str(predictions_targz_path)[:-7]
                    if not os.path.exists(str(predictions_path)):
                        print(f"Extracting predictions from tar.gz file into {predictions_path}")
                        with tarfile.open(predictions_targz_path, "r:gz") as tar:
                            if not args.dry_run:
                                tar.extractall(path=predictions_path)
                            else:
                                basename = os.path.basename(predictions_targz_path)[:-7]
                                tar.extractall(path=os.path.join(tmpdir, basename))
                                predictions_path = os.path.join(tmpdir, basename)
                                predictions_paths = [Path(predictions_path)]

                predictions_path = None
                for _ppath in predictions_paths:
                    if not _ppath.is_dir():
                        continue
                    predictions_path = _ppath
                    with (predictions_path / "info.json").open("r", encoding="utf8") as f:
                        info = json.load(f)
                        setval(info["dataset_metadata"], "scene", scene)
                        setval(info["dataset_metadata"], "id", dataset)
                        setval(info["render_dataset_metadata"], "scene", scene)
                        setval(info["render_dataset_metadata"], "id", dataset)
                    if has_change:
                        if not args.dry_run:
                            with (predictions_path / "info.json").open("w", encoding="utf8") as f:
                                json.dump(info, f, indent=2, ensure_ascii=False)
                        print(f"Fixed metadata for {predictions_path / 'info.json'}")

                # Compress back after changes
                if has_change:
                    for predictions_targz_path in scene_path.glob("predictions-*.tar.gz"):
                        if not args.dry_run:
                            print(f"Removing old tar.gz file {predictions_targz_path}")
                            predictions_targz_path.unlink()
                            print(f"Creating new tar.gz file for {predictions_path}")
                            assert predictions_path is not None, "Predictions path should not be None at this point."
                            os.system(f"tar -czf {predictions_targz_path} -C {predictions_path} .")


                for results_path in scene_path.glob("results-*.json"):
                    with results_path.open("r", encoding="utf8") as f:
                        results = json.load(f)
                        setval(results["render_dataset_metadata"], "scene", scene)
                        setval(results["render_dataset_metadata"], "id", dataset)
                        setval(results["nb_info"]["dataset_metadata"], "scene", scene)
                        setval(results["nb_info"]["dataset_metadata"], "id", dataset)
                    if has_change:
                        if not args.dry_run:
                            with results_path.open("w", encoding="utf8") as f:
                                json.dump(results, f, indent=2, ensure_ascii=False)
                        print(f"Fixed metadata for {results_path}")
                checkpoint_path = None
                for checkpoint_path in scene_path.glob("checkpoint-*"):
                    with (checkpoint_path / "nb-info.json").open("r", encoding="utf8") as f:
                        nb_info = json.load(f)
                        setval(nb_info["dataset_metadata"], "scene", scene)
                        setval(nb_info["dataset_metadata"], "id", dataset)
                    if has_change:
                        if not args.dry_run:
                            with (checkpoint_path / "nb-info.json").open("w", encoding="utf8") as f:
                                json.dump(nb_info, f, indent=2, ensure_ascii=False)
                        print(f"Fixed metadata for {checkpoint_path / 'nb-info.json'}")

                # Finally, we need to update the zip output files
                if results_path is None or checkpoint_path is None or predictions_path is None:
                    print(f"Skipping {scene_path} as it does not contain all necessary files.")
                    continue

                if has_change:
                    save_output_artifact(
                        checkpoint_path,
                        predictions_path,
                        results_path,
                        scene_path / "tensorboard",
                        scene_path / "output.zip" if not args.dry_run else os.path.join(tmpdir, "output.zip"),
                        validate=False,
                    )
                    print(f"Updated output.zip for {scene_path}")

