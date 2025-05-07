import os
import json
import logging
import click
import time
import tqdm
import numpy as np
from nerfbaselines import backends
from nerfbaselines import load_checkpoint
from nerfbaselines.datasets import dataset_index_select, load_dataset, dataset_load_features
from ._common import NerfBaselinesCliCommand, click_backend_option, IndicesClickType, Indices


@backends.run_on_host()
def _measure_render_times_local(method, cameras, output_names, *, num_repeats, num_repeats_median, warmup_steps=2):
    try:
        import torch
    except ImportError:
        torch = None

    times = []
    with tqdm.tqdm(total=len(cameras)*num_repeats*num_repeats_median+warmup_steps, desc="Measuring FPS") as pbar:
        for _ in range(warmup_steps):
            cam = cameras[0]
            with backends.zero_copy():
                method.render(cam, options={
                    "outputs": output_names.split(","),
                    "keep_torch": True,
                } if output_names is not None else {})
                pbar.update(1)

        for cam in cameras:
            sub_times = []
            for _ in range(num_repeats):
                for _ in range(num_repeats_median):
                    with backends.zero_copy():
                        if torch is not None:
                            torch.cuda.synchronize()
                        time_start = time.perf_counter()
                        method.render(cam, options={
                            "outputs": output_names.split(","),
                            "keep_torch": True,
                        } if output_names is not None else {})
                        if torch is not None:
                            torch.cuda.synchronize()
                        sub_times.append(time.perf_counter() - time_start)
                        pbar.update(1)
                times.append(float(np.median(sub_times)))
    
    # Return times
    return times


def _override_resolution(cameras, resolution_string: str):
    # Override resolution
    w, h = tuple(map(int, resolution_string.split("x")))
    for i in range(len(cameras)):
        oldw = cameras.image_sizes[i, 0]
        oldh = cameras.image_sizes[i, 1]
        aspect = oldw / oldh
        if w < 0:
            assert h > 0, "Either width or height must be positive"
            w = ((int(h * aspect) + abs(w) - 1) // abs(w)) * abs(w)
        elif h < 0:
            assert w > 0, "Either width or height must be positive"
            h = ((int(w / aspect) + abs(h) - 1) // abs(h)) * abs(h)

        # Rescale cameras
        cameras.intrinsics[i, 0] *= w / oldw
        cameras.intrinsics[i, 1] *= h / oldh
        cameras.intrinsics[i, 2] *= w / oldw
        cameras.intrinsics[i, 3] *= h / oldh
        cameras.image_sizes[i, 0] = w
        cameras.image_sizes[i, 1] = h


@click.command("measure-fps", cls=NerfBaselinesCliCommand, short_help="Measure FPS", help=("Measure FPS"))
@click.option("--checkpoint", default=None, required=True, type=str, help=(
    "Path to the checkpoint directory. It can also be a remote path (starting with `http(s)://`) or be a path inside a zip file."
))
@click.option("--data", type=str, required=True, help=(
    "A path to the dataset to render the cameras from. The dataset can be either an external dataset (e.g., a path starting with `external://{dataset}/{scene}`) or a local path to a dataset. If the dataset is an external dataset, the dataset will be downloaded and cached locally. If the dataset is a local path, the dataset will be loaded directly from the specified path."))
@click.option("--num-repeats", type=int, default=1, show_default=True, help="Number of times to repeat the rendering to estimate the FPS (results are averaged).")
@click.option("--num-repeats-median", type=int, default=1, show_default=True, help="Number of times to repeat the rendering to estimate the FPS (median is taken from results).")
@click.option("--split", type=str, default="test", show_default=True, help="Dataset split to use to estimate the FPS.")
@click.option("--data-indices", type=IndicesClickType(), default=Indices(slice(None, None)), help="Indices of the dataset to use to estimate the FPS. Default is to use all test cameras.")
@click.option("--resolution", type=str, default=None, help="Override the resolution of the output. Use 'widthxheight' format (e.g., 1920x1080). If one of the dimensions is negative, the aspect ratio will be preserved and the dimension will be rounded to the nearest multiple of the absolute value of the dimension.")
@click.option("--output-names", type=str, default="color", help="Comma separated list of output types (e.g. color,depth,accumulation). See the method's `get_info()['supported_outputs']` for supported outputs. By default, only `color` is rendered.")
@click.option("--output", type=str, default=None, help="Write output to a JSON file.")
@click_backend_option()
def measure_fps_command(*, checkpoint, backend_name, data, split="test", data_indices=None, resolution=None, num_repeats=1, num_repeats_median=1, output_names, output=None):
    # Load the checkpoint
    with load_checkpoint(checkpoint, backend=backend_name) as (model, _):

        # Load the dataset
        dataset = load_dataset(data, 
                               split=split, 
                               load_features=False)

        if data_indices is not None:
            dataset = dataset_index_select(dataset, list(data_indices.with_total(len(dataset["cameras"]))))

        # We load the features to fix image sizes
        if resolution is None:
            dataset = dataset_load_features(dataset)

        cameras = dataset["cameras"]

        # Override resolution if needed
        if resolution is not None:
            _override_resolution(cameras, resolution)

        # Measure FPS
        times = _measure_render_times_local(
            model, cameras, output_names, 
            num_repeats=num_repeats, 
            num_repeats_median=num_repeats_median)

        fps = len(times) / sum(times)
        logging.info(f"FPS: {fps:.2f}")

        if output is not None:
            assert output.endswith(".json"), "Output must be a JSON file"
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(output, "w", encoding="utf8") as f:
                json.dump({"fps": fps}, f, indent=2)
