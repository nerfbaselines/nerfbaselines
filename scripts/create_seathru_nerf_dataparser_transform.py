#!/usr/bin/env python3
import base64
import numpy as np
import json
import io
import argparse
from internal import configs, datasets
import sys
import os
import gin


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Path to the data directory')
parser.add_argument('--output', type=str, required=True, help='Path to the output dataparser_transform.json file')
parser.add_argument('--config', type=str, default='configs/llff_256_uw.gin')
args = parser.parse_args()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load config
import eval
root = os.path.dirname(os.path.abspath(eval.__file__))
gin.parse_config_file(os.path.join(root, args.config))
config = configs.load_config(save_config=False)
config.data_dir = args.data
mode = 'train' if config.eval_on_train else 'test'
dataset = datasets.load_dataset('test', config.data_dir, config)

def numpy_to_base64(array: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, array)
        return base64.b64encode(f.getvalue()).decode("ascii")

with open(args.output, "w+", encoding="utf8") as fp:
    fp.write(
        json.dumps(
            {
                "colmap_to_world_transform": dataset.colmap_to_world_transform.tolist(),
                "colmap_to_world_transform_base64": numpy_to_base64(dataset.colmap_to_world_transform),
                "pixtocam_ndc": dataset.pixtocam_ndc.tolist(),
                "pixtocam_ndc_base64": numpy_to_base64(dataset.pixtocam_ndc),
            }, indent=2
        )
    )
