import importlib
import contextlib
import subprocess
import shutil
import struct
import sys
import json
import hashlib
import time
import os
import math
import logging
from pathlib import Path
from typing import Optional, Union, List, Any, Dict, Tuple, cast, FrozenSet, Callable, Sequence, Set
import tqdm.contrib.logging
import numpy as np
from PIL import Image
import nerfbaselines
from . import (
    Method, 
    MethodSpec,
    EvaluationProtocol, 
    __version__, 
    Dataset,
)
from .backends import run_on_host
from .io import (
    deserialize_nb_info, 
    serialize_nb_info, 
    save_output_artifact,
    new_nb_info,
)
from ._registry import loggers_registry
from .utils import (
    Indices, 
    image_to_srgb, 
    visualize_depth,
    convert_image_dtype,
)
from .datasets import dataset_index_select
from .evaluation import (
    render_all_images, evaluate, build_evaluation_protocol,
)
from .logging import ConcatLogger, Logger, log_metrics
try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict


MetricAccumulationMode = Literal["average", "last", "sum"]


class ResourcesUtilizationInfo(TypedDict, total=False):
    memory: int
    gpu_memory: int
    gpu_name: str


@run_on_host()
def get_resources_utilization_info(pid: Optional[int] = None) -> ResourcesUtilizationInfo:
    import platform

    if pid is None:
        pid = os.getpid()

    info: ResourcesUtilizationInfo = {}

    # Get all cpu memory and running processes
    all_processes = set((pid,))
    current_platform = platform.system()

    if current_platform == "Windows":  # Windows
        def get_memory_usage_windows(pid: int) -> int:
            try:
                process = subprocess.Popen(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    stdout=subprocess.PIPE,
                    text=True
                )
                out, _ = process.communicate()
                mem_usage_str = out.strip().split(',')[4].strip('"').replace(' K', '').replace(',', '')
                return int(mem_usage_str) * 1024  # Convert KB to bytes
            except Exception:
                return 0
        try:
            mem = get_memory_usage_windows(pid)
            all_processes = set((pid,))
            out = subprocess.check_output(
                ["wmic", "process", "where", f"(ParentProcessId={pid})", "get", "ProcessId"],
                text=True
            )
            children_pids = [int(line.strip()) for line in out.strip().split() if line.strip().isdigit()]
            for child_pid in children_pids:
                mem += get_memory_usage_windows(child_pid)
                all_processes.add(child_pid)
            info["memory"] = (mem + 1024 - 1) // 1024
        except Exception:
            logging.error(f"Failed to get resource usage information on {current_platform}", exc_info=True)
            return info
    else:  # Linux or macOS
        try:
            mem = 0
            out = subprocess.check_output("ps -ax -o pid= -o ppid= -o rss=".split(), text=True).splitlines()
            mem_used: Dict[int, int] = {}
            children = {}
            for line in out:
                cpid, ppid, used_memory = map(int, line.split())
                mem_used[cpid] = used_memory
                children.setdefault(ppid, set()).add(cpid)
            all_processes = set()
            stack = [pid]
            while stack:
                cpid = stack.pop()
                all_processes.add(cpid)
                mem += mem_used[cpid]
                stack.extend(children.get(cpid, []))
            info["memory"] = (mem + 1024 - 1) // 1024
        except Exception:
            logging.error(f"Failed to get resource usage information on {current_platform}", exc_info=True)
            return info

    try:
        gpu_memory = 0
        gpus = {}
        uuids = set()
        nvidia_smi_command = "nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid,gpu_name --format=csv,noheader,nounits"
        if current_platform == "Windows":
            out = subprocess.check_output("nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid,gpu_name --format=csv,noheader,nounits", shell=True, text=True).splitlines()
        else:
            out = subprocess.check_output(nvidia_smi_command.split(), text=True).splitlines()
        for line in out:
            cpid, used_memory, uuid, gpu_name = tuple(x.strip() for x in line.split(",", 3))
            try:
                cpid = int(cpid)
                used_memory = int(used_memory)
            except ValueError:
                # Unused GPUs could sometimes return [N/A]
                continue
            if cpid in all_processes:
                gpu_memory += used_memory
                if uuid not in uuids:
                    uuids.add(uuid)
                    gpus[gpu_name] = gpus.get(gpu_name, 0) + 1
        info["gpu_name"] = ",".join(f"{k}:{v}" if v > 1 else k for k, v in gpus.items())
        info["gpu_memory"] = gpu_memory
    except Exception:
        logging.error(f"Failed to get GPU utilization on {current_platform}", exc_info=True)
        return info

    return info


def _get_config_overrides_from_presets(spec: MethodSpec, presets: Union[Set[str], Sequence[str]]) -> Dict[str, Any]:
    """
    Apply presets to a method spec and return the config overrides.

    Args:
        spec: Method spec
        presets: List of presets to apply

    Returns:
        A dictionary of config overrides
    """
    _config_overrides = {}
    _presets = set(presets)
    for preset_name, preset in spec.get("presets", {}).items():
        if preset_name not in _presets:
            continue
        _config_overrides.update({
            k: v for k, v in preset.items()
            if not k.startswith("@")
        })
    return _config_overrides


def _get_presets_to_apply(spec: MethodSpec, dataset_metadata: Dict[str, Any], presets: Union[Set[str], Sequence[str], None] = None) -> Set[str]:
    """
    Given a method spec, dataset metadata, and the optional list of presets from the user,
    this function returns the list of presets that should be applied.

    Args:
        spec: Method spec
        dataset_metadata: Dataset metadata
        presets: List of presets to apply or a special "@auto" preset that will automatically apply presets based on the dataset metadata

    Returns:
        List of presets to apply
    """
    # Validate presets for MethodSpec
    auto_presets = presets is None
    _presets = set(presets or ())
    _condition_data = dataset_metadata.copy()
    _condition_data["dataset"] = _condition_data.pop("id", "")

    for preset in presets or []:
        if preset == "@auto":
            if auto_presets:
                raise ValueError("Cannot specify @auto preset multiple times")
            auto_presets = True
            _presets.remove("@auto")
            continue
        if preset not in spec.get("presets", {}):
            raise ValueError(f"Preset {preset} not found in method spec {spec['id']}. Available presets: {','.join(spec.get('presets', {}).keys())}")
    if auto_presets:
        for preset_name, preset in spec.get("presets", {}).items():
            apply = preset.get("@apply", [])
            if not apply:
                continue
            for condition in apply:
                if all(_condition_data.get(k, "") == v for k, v in condition.items()):
                    _presets.add(preset_name)
    return _presets


def make_image_grid(*images: np.ndarray, ncol=None, padding=2, max_width=1920, background: Union[None, Tuple[float, float, float], np.ndarray] = None):
    if ncol is None:
        ncol = len(images)
    dtype = images[0].dtype
    if background is None:
        background = np.full((3,), 255 if dtype == np.uint8 else 1, dtype=dtype)
    elif isinstance(background, tuple):
        background = np.array(background, dtype=dtype)
    elif isinstance(background, np.ndarray):
        background = convert_image_dtype(background, dtype=dtype)
    else:
        raise ValueError(f"Invalid background type {type(background)}")
    nrow = int(math.ceil(len(images) / ncol))
    scale_factor = 1
    height, width = tuple(map(int, np.max([x.shape[:2] for x in images], axis=0).tolist()))
    if max_width is not None:
        scale_factor = min(1, (max_width - padding * (ncol - 1)) / (ncol * width))
        height = int(height * scale_factor)
        width = int(width * scale_factor)

    def interpolate(image) -> np.ndarray:
        img = Image.fromarray(image)
        img_width, img_height = img.size
        aspect = img_width / img_height
        img_width = int(min(width, aspect * height))
        img_height = int(img_width / aspect)
        img = img.resize((img_width, img_height))
        return np.array(img)

    images = tuple(map(interpolate, images))
    grid: np.ndarray = np.ndarray(
        (height * nrow + padding * (nrow - 1), width * ncol + padding * (ncol - 1), images[0].shape[-1]),
        dtype=dtype,
    )
    grid[..., :] = background
    for i, image in enumerate(images):
        x = i % ncol
        y = i // ncol
        h, w = image.shape[:2]
        offx = x * (width + padding) + (width - w) // 2
        offy = y * (height + padding) + (height - h) // 2
        grid[offy : offy + h, 
             offx : offx + w] = image
    return grid


class MetricsAccumulator:
    def __init__(
        self,
        options: Optional[Dict[str, MetricAccumulationMode]] = None,
    ):
        self.options = options or {}
        self._state = None

    def update(self, metrics: Dict[str, Union[int, float]]) -> None:
        if self._state is None:
            self._state = {}
        state = self._state
        n_iters_since_update = state["n_iters_since_update"] = state.get("n_iters_since_update", {})
        for k, v in metrics.items():
            accumulation_mode = self.options.get(k, "average")
            n_iters_since_update[k] = n = n_iters_since_update.get(k, 0) + 1
            if k not in state:
                state[k] = 0
            if accumulation_mode == "last":
                state[k] = v
            elif accumulation_mode == "average":
                state[k] = state[k] * ((n - 1) / n) + v / n
            elif accumulation_mode == "sum":
                state[k] += v
            else:
                raise ValueError(f"Unknown accumulation mode {accumulation_mode}")

    def pop(self) -> Dict[str, Union[int, float]]:
        if self._state is None:
            return {}
        state = self._state
        self._state = None
        state.pop("n_iters_since_update", None)
        return state




def eval_few(method: Method, logger: Logger, dataset: Dataset, *, split: str, step, evaluation_protocol: EvaluationProtocol):
    rand_number, = struct.unpack("<Q", hashlib.sha1(str(step).encode("utf8")).digest()[:8])

    idx = rand_number % len(dataset["image_paths"])
    dataset_slice = dataset_index_select(dataset, [idx])
    images = dataset_slice["images"]
    image_sizes = dataset_slice["cameras"].image_sizes
    total_rays = image_sizes.prod(-1).sum()

    expected_scene_scale: Optional[float] = dataset_slice["metadata"].get("expected_scene_scale")

    start = time.perf_counter()
    # Pseudo-randomly select an image based on the step
    logging.info(f"Rendering single {split} image at step={step}")
    predictions = evaluation_protocol.render(method, dataset_slice)
    elapsed = time.perf_counter() - start

    _metrics = evaluation_protocol.evaluate(predictions, dataset_slice)
    metrics = cast(Dict[str, Union[float, int, str]], _metrics)
    del _metrics
    w, h = dataset_slice["cameras"].image_sizes[0]
    gt = images[0][:h, :w]
    color = predictions["color"]

    # Print metrics to console
    metric_to_show = None
    if "psnr" in metrics:
        metric_to_show = "psnr"
    elif "loss" in metrics:
        metric_to_show = "loss"
    if metric_to_show is not None:
        logging.info(f"Evaluated single {split} image at step={step}, {metric_to_show}={metrics[metric_to_show]:.4f}")

    background_color = dataset_slice["metadata"].get("background_color", None)
    dataset_colorspace = dataset_slice["metadata"].get("color_space", "srgb")
    color_srgb = image_to_srgb(color, np.uint8, color_space=dataset_colorspace, background_color=background_color)
    gt_srgb = image_to_srgb(gt, np.uint8, color_space=dataset_colorspace, background_color=background_color)

    image_path = dataset_slice["image_paths"][0]
    images_root = dataset_slice.get("image_paths_root")
    if images_root is not None:
        if str(image_path).startswith(str(images_root)):
            image_path = str(Path(image_path).relative_to(images_root))

    metrics["image-path"] = image_path
    metrics["fps"] = 1 / elapsed
    metrics["rays-per-second"] = total_rays / elapsed
    metrics["time"] = elapsed

    # Add depth
    depth = None
    if "depth" in predictions:
        near_far = dataset_slice["cameras"].nears_fars[0] if dataset_slice["cameras"].nears_fars is not None else None
        depth = visualize_depth(predictions["depth"], expected_scale=expected_scene_scale, near_far=near_far)

    # Log to loggers
    if logger:
        logging.debug(f"logging image to {logger}")

        log_metrics(logger, metrics, prefix=f"eval-few-{split}/", step=step)

        color_vis = make_image_grid(gt_srgb, color_srgb)
        with logger.add_event(step) as event:
            event.add_image(
                f"eval-few-{split}/color",
                color_vis,
                display_name=image_path,
                description="left: gt, right: prediction",
            )
            if depth is not None:
                event.add_image(
                    f"eval-few-{split}/depth",
                    depth,
                    display_name=image_path,
                    description="depth",
                )


def eval_all(method: Method, logger: Optional[Logger], dataset: Dataset, *, output: str, step: int, evaluation_protocol: EvaluationProtocol, split: str, nb_info):
    metrics: Optional[Dict[str, float]] = {} if logger else None
    expected_scene_scale = dataset["metadata"].get("expected_scene_scale")

    # Store predictions, compute metrics, etc.
    prefix = dataset["image_paths_root"]
    if prefix is None:
        prefix = Path(os.path.commonpath(dataset["image_paths"]))

    if split != "test":
        output_metrics = os.path.join(output, f"results-{step}-{split}.json")
        output = os.path.join(output, f"predictions-{step}-{split}.tar.gz")
    else:
        output_metrics = os.path.join(output, f"results-{step}.json")
        output = os.path.join(output, f"predictions-{step}.tar.gz")

    if os.path.exists(output):
        if os.path.isfile(output):
            os.unlink(output)
        else:
            shutil.rmtree(output)
        logging.warning(f"Removed existing predictions at {output}")

    if os.path.exists(output_metrics):
        os.unlink(output_metrics)
        logging.warning(f"Removed existing results at {output_metrics}")

    start = time.perf_counter()
    num_vis_images = 16
    vis_images: List[Tuple[np.ndarray, np.ndarray]] = []
    vis_depth: List[np.ndarray] = []
    image_sizes = dataset["cameras"].image_sizes
    total_rays = image_sizes.prod(-1).sum()
    assert image_sizes is not None
    for (i, gt), pred, (w, h) in zip(
        enumerate(dataset["images"]),
        render_all_images(
            method,
            dataset,
            output=output,
            description=f"Rendering all images at step={step}",
            nb_info=nb_info,
            evaluation_protocol=evaluation_protocol,
        ),
        image_sizes,
    ):
        if len(vis_images) < num_vis_images:
            color = pred["color"]
            background_color = dataset["metadata"].get("background_color", None)
            dataset_colorspace = dataset["metadata"].get("color_space", "srgb")
            color_srgb = image_to_srgb(color, np.uint8, color_space=dataset_colorspace, background_color=background_color)
            gt_srgb = image_to_srgb(gt[:h, :w], np.uint8, color_space=dataset_colorspace, background_color=background_color)
            vis_images.append((gt_srgb, color_srgb))
            if "depth" in pred:
                near_far = dataset["cameras"].nears_fars[i] if dataset["cameras"].nears_fars is not None else None
                vis_depth.append(visualize_depth(pred["depth"], expected_scale=expected_scene_scale, near_far=near_far))
    elapsed = time.perf_counter() - start

    # Compute metrics
    info = evaluate(
        output, 
        output_metrics, 
        evaluation_protocol=evaluation_protocol,
        description=f"evaluating all images at step={step}")
    metrics = info["metrics"]

    if logger:
        assert metrics is not None, "metrics must be computed"
        logging.debug(f"logging metrics to {logger}")
        metrics["fps"] = len(dataset["image_paths"]) / elapsed
        metrics["rays-per-second"] = total_rays / elapsed
        metrics["time"] = elapsed
        log_metrics(logger, metrics, prefix=f"eval-all-{split}/", step=step)

        num_cols = int(math.sqrt(len(vis_images)))

        color_vis = make_image_grid(
            make_image_grid(*[x[0] for x in vis_images], ncol=num_cols),
            make_image_grid(*[x[1] for x in vis_images], ncol=num_cols),
        )

        logger.add_image(f"eval-all-{split}/color", 
                         color_vis, 
                         display_name="color", 
                         description="left: gt, right: prediction", 
                         step=step)
    return metrics


def build_logger(loggers: FrozenSet[str]) -> Callable[[str], Logger]:
    """
    Validates the list of loggers and builds a logger object. It returns a lazy function
    that initializes the logger when called.

    Args:
        loggers: Set of loggers to use

    Returns:
        A function that initializes the logger when called. It takes the output directory as it's argument
    """

    # Validate loggers
    for logger in loggers:
        if logger not in loggers_registry:
            raise ValueError(f"Unknown logger {logger}")

    def build(output: str) -> Logger:
        _loggers = []
        for logger in loggers:
            spec = nerfbaselines.get_logger_spec(logger)
            package, class_name = spec["logger_class"].split(":", 1)
            logger_cls: Any = importlib.import_module(package)
            for part in class_name.split("."):
                logger_cls = getattr(logger_cls, part)
            _loggers.append(logger_cls(output))
        logging.info("Initialized loggers: " + ",".join(loggers))
        return ConcatLogger(_loggers)
    return build


def _is_tensorboard_enabled(logger: Logger, output: str) -> bool:
    from nerfbaselines.logging import TensorboardLogger, ConcatLogger
    if isinstance(logger, TensorboardLogger):
        if os.path.abspath(logger._output) == os.path.abspath(output):
            return True
    elif isinstance(logger, ConcatLogger):
        return any(_is_tensorboard_enabled(sub_logger, output) for sub_logger in logger.loggers)
    return False


def get_presets_and_config_overrides(method_spec: MethodSpec, dataset_metadata: Dict, *, presets=None, config_overrides=None):
    """
    Given a method spec, dataset metadata, and the optional list of presets from the user,
    this function computes the list of presets that should be applied. The presets
    are then applied to obtain config_overrides which are merged with the provided config_overrides.

    Args:
        method_spec: Method spec
        dataset_metadata: Dataset metadata
        presets: List of presets to apply or a special "@auto" preset that will automatically apply presets based on the dataset metadata
        config_overrides: Config overrides to be applied after all presets were processed

    Returns:
        Tuple: List of applied presets, final config overrides
    """
    # Apply config overrides for the train dataset
    if dataset_metadata.get("id") is None:
        logging.warning("Dataset ID not specified, dataset-specific config overrides may not be applied")

    _presets = _get_presets_to_apply(method_spec, dataset_metadata, presets)
    dataset_overrides = _get_config_overrides_from_presets(
        method_spec,
        _presets,
    )
    if dataset_overrides is not None:
        dataset_overrides = dataset_overrides.copy()
        dataset_overrides.update(config_overrides or {})
        config_overrides = dataset_overrides
    del dataset_overrides
    return _presets, config_overrides


class Trainer:
    def __init__(
        self,
        *,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        method: Method,
        output: str = ".",
        save_iters: Indices = Indices.every_iters(10_000, zero=True),
        eval_few_iters: Indices = Indices.every_iters(2_000),
        eval_all_iters: Indices = Indices([-1]),
        logger: Union[Callable[[str], Logger], Logger, None] = None,
        generate_output_artifact: Optional[bool] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        applied_presets: Optional[FrozenSet[str]] = None,
    ):
        self._num_iterations = 0
        self.method = method
        self.model_info = self.method.get_info()
        self.test_dataset: Optional[Dataset] = test_dataset

        
        self.step = self.model_info.get("loaded_step") or 0
        if self.num_iterations is None:
            raise RuntimeError(f"Method {self.model_info['method_id']} must specify the default number of iterations")

        self.output = output
        logging.info(f"Output directory: {output}")

        self.save_iters = save_iters
        self.eval_few_iters = eval_few_iters
        self.eval_all_iters = eval_all_iters
        self.generate_output_artifact = generate_output_artifact
        self.config_overrides = config_overrides
        if logger is not None and not callable(logger):
            logger = cast(Callable[[str], Logger], lambda _: logger)
        self._logger_fn = logger

        self._applied_presets = applied_presets
        self._logger: Optional[Logger] = None
        self._average_image_size = None
        self._dataset_metadata = None
        self._total_train_time = 0
        self._resources_utilization_info = None
        self._train_dataset_for_eval = None
        self._acc_metrics = MetricsAccumulator({
            "total-train-time": "last",
            "learning-rate": "last",
        })

        # Update schedules
        self.num_iterations = self.model_info["num_iterations"]

        # Restore checkpoint if specified
        loaded_checkpoint = self.model_info.get("loaded_checkpoint")
        if loaded_checkpoint is not None:
            with Path(loaded_checkpoint).joinpath("nb-info.json").open("r", encoding="utf8") as f:
                info = json.load(f)
                info = deserialize_nb_info(info)
                self._total_train_time = info["total_train_time"]
                self._resources_utilization_info = info["resources_utilization"]

        # Validate and setup datasets
        self._setup_data(train_dataset, test_dataset)

    @property
    def num_iterations(self):
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value):
        self._num_iterations = value

        # Fix total for indices
        for v in vars(self).values():
            if isinstance(v, Indices):
                v.total = self.num_iterations + 1

    def _setup_data(self, train_dataset: Dataset, test_dataset: Optional[Dataset]):
        # Validate and setup datasets
        # Store a slice of train dataset used for eval_few
        self._average_image_size = train_dataset["cameras"].image_sizes.prod(-1).astype(np.float32).mean()
        dataset_background_color = train_dataset["metadata"].get("background_color")
        if dataset_background_color is not None:
            assert isinstance(dataset_background_color, np.ndarray), "Dataset background color must be a numpy array"
            assert dataset_background_color.dtype == np.uint8, "Dataset background color must be an uint8 array"
        train_dataset_indices = np.linspace(0, len(train_dataset["image_paths"]) - 1, 16, dtype=int)
        self._train_dataset_for_eval = dataset_index_select(train_dataset, train_dataset_indices)

        color_space = train_dataset["metadata"].get("color_space")
        assert color_space is not None
        self._dataset_metadata = train_dataset["metadata"].copy()

        # Setup test dataset dataset
        self.test_dataset = test_dataset
        if test_dataset is not None:
            if test_dataset["metadata"].get("color_space") != color_space:
                raise RuntimeError(f"train dataset color space {color_space} != test dataset color space {test_dataset['metadata'].get('color_space')}")
            test_background_color = test_dataset["metadata"].get("background_color")
            if test_background_color is not None:
                assert isinstance(test_background_color, np.ndarray), "Dataset's background_color must be a numpy array"
            if not (
                (test_background_color is None and dataset_background_color is None) or 
                (
                    test_background_color is not None and 
                    dataset_background_color is not None and
                    np.array_equal(test_background_color, dataset_background_color)
                )
            ):
                raise RuntimeError(f"train dataset color space {dataset_background_color} != test dataset color space {test_dataset['metadata'].get('background_color')}")

        self._validate_output_artifact()
        self._evaluation_protocol = build_evaluation_protocol(self._dataset_metadata["evaluation_protocol"])

    def _validate_output_artifact(self):
        # Validate generate output artifact
        if self.generate_output_artifact is None or self.generate_output_artifact:
            messages = []
            # Model is saved automatically at the end!
            # if self.num_iterations not in self.save_iters:
            #     messages.append(f"num_iterations ({self.num_iterations}) must be in save_iters: {self.save_iters}")
            if self.num_iterations not in self.eval_all_iters:
                messages.append(f"num_iterations ({self.num_iterations}) must be in eval_all_iters: {self.eval_all_iters}")
            if self.generate_output_artifact is None and messages:
                logging.warning("Disabling output artifact generation due to the following problems:")
                for message in messages:
                    logging.warning(message)
                self.generate_output_artifact = False
            elif messages:
                logging.error("Cannot generate output artifact due to the following problems:")
                for message in messages:
                    logging.error(message)
                sys.exit(1)

    def _get_nb_info(self):
        assert self._dataset_metadata is not None, "dataset_metadata must be set"
        return new_nb_info(
            self._dataset_metadata,
            self.method,
            self.config_overrides,
            evaluation_protocol=self._evaluation_protocol,
            resources_utilization_info=self._resources_utilization_info,
            total_train_time=self._total_train_time,
            applied_presets=self._applied_presets,
        )

    def save(self):
        path = os.path.join(self.output, f"checkpoint-{self.step}")  # pyright: ignore[reportCallIssue]
        os.makedirs(os.path.join(self.output, f"checkpoint-{self.step}"), exist_ok=True)
        self.method.save(str(path))
        with open(os.path.join(path, "nb-info.json"), mode="w+", encoding="utf8") as f:
            json.dump(serialize_nb_info(self._get_nb_info()), f, indent=2)
        logging.info(f"checkpoint saved at step={self.step}")

    def train_iteration(self):
        start = time.perf_counter()
        metrics = self.method.train_iteration(self.step)

        elapsed = time.perf_counter() - start
        self._total_train_time += elapsed

        # Replace underscores with dashes for in metrics
        metrics = {k.replace("_", "-"): v for k, v in metrics.items()}
        metrics["time"] = elapsed
        metrics["total-train-time"] = self._total_train_time
        if "num_rays" in metrics:
            batch_size = metrics.pop("num_rays")
            metrics["rays-per-second"] = batch_size / elapsed
            if self._average_image_size is not None:
                metrics["fps"] = batch_size / elapsed / self._average_image_size
        return metrics

    def get_logger(self) -> Logger:
        if self._logger is None:
            if self._logger_fn is None:
                self._logger = ConcatLogger([])
            else:
                self._logger = self._logger_fn(self.output)

            # After the loggers are initialized, we perform one last check for the output artifacts
            if self.generate_output_artifact is None or self.generate_output_artifact:
                if not _is_tensorboard_enabled(self._logger, os.path.join(self.output, "tensorboard")):
                    logging.error("Add tensorboard logger in order to produce output artifact. Please add `--vis tensorboard` to the command line arguments. Or disable output artifact generation with `--no-output-artifact`")
                    if self.generate_output_artifact is None:
                        self.generate_output_artifact = False
                    else:
                        sys.exit(1)
                else:
                    self.generate_output_artifact = True
        return self._logger

    def _update_resource_utilization_info(self):
        update = False
        util: Dict[str, Union[int, float]] = {}
        if self._resources_utilization_info is None:
            update = True
        elif self.step % 1000 == 11:
            update = True
            util = self._resources_utilization_info
        if update:
            logging.debug(f"Computing resource utilization at step={self.step}")
            new_util = cast(Dict[str, int], get_resources_utilization_info())
            for k, v in new_util.items():
                if k not in util:
                    util[k] = 0
                if isinstance(v, str):
                    util[k] = v
                else:
                    util[k] = max(util[k], v)
            self._resources_utilization_info = util

    def train(self):
        assert self.num_iterations is not None, "num_iterations must be set"
        assert self._average_image_size is not None, "dataset not set"
        if self.step == 0 and self.step in self.save_iters:
            self.save()

        # Initialize loggers before training loop for better tqdm output
        logger = self.get_logger()

        update_frequency = 100
        final_metrics = None
        with tqdm.contrib.logging.tqdm_logging_redirect(total=self.num_iterations, initial=self.step, desc="training") as pbar:
            for i in range(self.step, self.num_iterations):
                final_metrics = None
                self.step = i
                metrics = self.train_iteration()
                # Checkpoint changed, reset sha
                self._checkpoint_sha = None
                self.step = i + 1
                pbar.update()

                # Update accumulated metrics
                self._acc_metrics.update(metrics)

                # Update resource utilization info
                self._update_resource_utilization_info()

                # Log metrics and update progress bar
                if self.step % update_frequency == 0 or self.step == self.num_iterations:
                    acc_metrics = self._acc_metrics.pop()
                    postfix = {}
                    if "psnr" in acc_metrics:
                        postfix["train/psnr"] = f'{acc_metrics["psnr"]:.4f}'
                    elif "loss" in acc_metrics:
                        postfix["train/loss"] = f'{acc_metrics["loss"]:.4f}'
                    if postfix:
                        pbar.set_postfix(postfix)
                    log_metrics(logger, acc_metrics, prefix="train/", step=self.step)

                # Visualize and save
                if self.step in self.save_iters:
                    self.save()
                if self.step in self.eval_few_iters:
                    self.eval_few()
                if self.step in self.eval_all_iters:
                    final_metrics = self.eval_all()

        # We can print the results because the evaluation was run for the last step
        if final_metrics is not None:
            logging.info("Final evaluation results:\n" + 
                         "\n".join(f"   {k.replace('_','-')}: {v:.4f}" for k, v in final_metrics.items()))

        # Save if not saved by default
        if self.step not in self.save_iters:
            self.save()

        # Generate output artifact if enabled
        if self.generate_output_artifact:
            save_output_artifact(
                Path(self.output) / f"checkpoint-{self.step}",
                Path(self.output) / f"predictions-{self.step}.tar.gz",
                Path(self.output) / f"results-{self.step}.json",
                Path(self.output) / "tensorboard",
                Path(self.output) / "output.zip",
                validate=False,
            )

    def eval_all(self):
        if self.test_dataset is None:
            logging.warning("Skipping eval_all on test dataset - no test dataset")
            return
        logger = self.get_logger()
        nb_info = self._get_nb_info()
        return eval_all(self.method, logger, self.test_dataset, 
                        step=self.step, evaluation_protocol=self._evaluation_protocol,
                        split="test", nb_info=nb_info, output=self.output)

    def eval_few(self):
        logger = self.get_logger()

        assert self._train_dataset_for_eval is not None, "train_dataset_for_eval must be set"
        rand_number, = struct.unpack("<Q", hashlib.sha1(str(self.step).encode("utf8")).digest()[:8])

        idx = rand_number % len(self._train_dataset_for_eval["image_paths"])
        dataset_slice = dataset_index_select(self._train_dataset_for_eval, slice(idx, idx + 1))

        eval_few(self.method, logger, dataset_slice, split="train", step=self.step, evaluation_protocol=self._evaluation_protocol)
        
        if self.test_dataset is None:
            logging.warning("Skipping eval_few on test dataset - no eval dataset")
            return

        idx = rand_number % len(self.test_dataset["image_paths"])
        dataset_slice = dataset_index_select(self.test_dataset, slice(idx, idx + 1))
        eval_few(self.method, logger, dataset_slice, split="test", step=self.step, evaluation_protocol=self._evaluation_protocol)
