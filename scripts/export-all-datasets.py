#!/usr/bin/env python3
import json
import fnmatch
import os
import argparse
import nerfbaselines
from nerfbaselines.datasets import load_dataset
from nerfbaselines.viewer._static import export_viewer_dataset
from nerfbaselines.viewer import write_dataset_pointcloud


parser = argparse.ArgumentParser(description='Export all datasets.')
parser.add_argument('--output', type=str, help='The output directory.')
parser.add_argument('--filter', type=str, help='Filter datasets by name.')
args = parser.parse_args()


# First, get list of all available datasets
datasets = {}
for dataset_id in nerfbaselines.get_supported_datasets():
    spec = nerfbaselines.get_dataset_spec(dataset_id)
    scenes = spec.get("metadata", {}).get("scenes", [])
    if not scenes:
        continue
    for scene in scenes:
        datasets[f"{dataset_id}/{scene['id']}"] = {
            "dataset": dataset_id,
            "scene": scene["id"],
        }
keys = list(datasets.keys())
keys.sort()
keys = fnmatch.filter(keys, args.filter) if args.filter else keys

for k in keys:
    if os.path.exists(os.path.join(args.output, k+"-nbv.json")):
        print(f"Skipping {k}")
        continue
    train_dataset = load_dataset(f"external://{k}", "train", features=("points3D_xyz", "points3D_rgb"))
    test_dataset = load_dataset(f"external://{k}", "test", features=("points3D_xyz", "points3D_rgb"))
    dataset = export_viewer_dataset(train_dataset, test_dataset)
        
    pardir = os.path.dirname(os.path.abspath(os.path.join(args.output, k)))
    os.makedirs(pardir, exist_ok=True)

    # Generate ply file
    if train_dataset.get("points3D_xyz") is not None:
        with open(os.path.join(args.output, k+"-pointcloud.ply"), "wb") as f:
            write_dataset_pointcloud(f, train_dataset["points3D_xyz"], train_dataset["points3D_rgb"])

    dataset["pointcloud_url"] = "./" + k.split("/")[-1]+"-pointcloud.ply"
    with open(os.path.join(args.output, k+"-nbv.json"), "w") as f:
        f.write(json.dumps(dataset, indent=2))
