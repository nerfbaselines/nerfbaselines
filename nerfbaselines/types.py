from abc import abstractmethod
from typing import Optional, Callable, Iterable, List, Dict, TYPE_CHECKING, Any, cast, Union
from dataclasses import dataclass, field
import dataclasses
import os
from pathlib import Path
import numpy as np

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore
try:
    from typing import runtime_checkable
except ImportError:
    from typing_extensions import runtime_checkable  # type: ignore
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore
try:
    from typing import get_args, get_origin
except ImportError:
    from typing_extensions import get_args, get_origin  # noqa: F401
try:
    from typing import NotRequired  # type: ignore
except ImportError:
    from typing_extensions import NotRequired  # type: ignore
try:
    from typing import FrozenSet
except ImportError:
    from typing_extensions import FrozenSet  # type: ignore
from .cameras import Cameras, CameraModel
from .utils import mark_host, padded_stack, generate_interface


ColorSpace = Literal["srgb", "linear"]
NB_PREFIX = os.path.expanduser(os.environ.get("NERFBASELINES_PREFIX", "~/.cache/nerfbaselines"))


@dataclass
class Dataset:
    cameras: Cameras  # [N]

    file_paths: List[str]
    sampling_mask_paths: Optional[List[str]] = None
    file_paths_root: Optional[Path] = None

    images: Optional[Union[np.ndarray, List[np.ndarray]]] = None  # [N][H, W, 3]
    sampling_masks: Optional[Union[np.ndarray, List[np.ndarray]]] = None  # [N][H, W]
    points3D_xyz: Optional[np.ndarray] = None  # [M, 3]
    points3D_rgb: Optional[np.ndarray] = None  # [M, 3]

    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.file_paths_root is None:
            self.file_paths_root = Path(os.path.commonpath(self.file_paths))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, i) -> "Dataset":
        assert isinstance(i, (slice, int, np.ndarray))

        def index(obj):
            if obj is None:
                return None
            if isinstance(obj, Cameras):
                if len(obj) == 1:
                    return obj if isinstance(i, int) else obj
                return obj[i]
            if isinstance(obj, np.ndarray):
                if obj.shape[0] == 1:
                    return obj[0] if isinstance(i, int) else obj
                obj = obj[i]
                return obj
            if isinstance(obj, list):
                indices = np.arange(len(self))[i]
                if indices.ndim == 0:
                    return obj[indices]
                return [obj[i] for i in indices]
            raise ValueError(f"Cannot index object of type {type(obj)}")

        return dataclasses.replace(self, **{k: index(v) for k, v in self.__dict__.items() if k not in {"file_paths_root", "points3D_xyz", "points3D_rgb", "metadata"}})

    @mark_host
    def load_features(self, required_features, supported_camera_models=None):
        # Import lazily here because the Dataset class
        # may be used in places where some of the dependencies
        # are not available.
        from .datasets._common import dataset_load_features

        dataset_load_features(self, required_features, supported_camera_models)
        return self

    @property
    def expected_scene_scale(self):
        return self.metadata.get("expected_scene_scale")

    @property
    def color_space(self) -> ColorSpace:
        color_space = self.metadata.get("color_space", None)
        return "srgb" if color_space is None else color_space

    def clone(self):
        return dataclasses.replace(
            self, 
            cameras=self.cameras.clone(), 
            images=self.images.copy() if self.images is not None else None,
            metadata=self.metadata.copy(),
            sampling_masks=self.sampling_masks.copy() if self.sampling_masks is not None else None)


@dataclass(frozen=True)
class CurrentProgress:
    i: int
    total: int
    image_i: int
    image_total: int


ProgressCallback = Callable[[CurrentProgress], None]


def batched(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i : i + batch_size]


DatasetFeature = Literal["color", "points3D_xyz", "points3D_rgb"]


if TypedDict is not None:
    class RenderOutput(TypedDict):
        color: np.ndarray  # [h w 3]
        depth: NotRequired[np.ndarray]  # [h w]
else:
    RenderOutput = Dict


@dataclass
class MethodInfo:
    name: str
    required_features: FrozenSet[DatasetFeature] = field(default_factory=frozenset)
    supported_camera_models: FrozenSet = field(default_factory=lambda: frozenset((CameraModel.PINHOLE,)))


@dataclass
class ModelInfo:
    name: str
    loaded_step: Optional[int] = None
    loaded_checkpoint: Optional[str] = None
    num_iterations: Optional[int] = None
    batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None
    required_features: FrozenSet[DatasetFeature] = field(default_factory=frozenset)
    supported_camera_models: FrozenSet = field(default_factory=lambda: frozenset((CameraModel.PINHOLE,)))
    hparams: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
@generate_interface
class Method(Protocol):
    def __init__(self, 
                 *,
                 checkpoint: Union[Path, None] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[Dict[str, Any]] = None):
        pass

    @classmethod
    def install(self):
        """
        Install the method.
        """
        pass

    @classmethod
    @abstractmethod
    def get_method_info(cls) -> MethodInfo:
        """
        Get method info needed to initialize the datasets.

        Returns:
            Method info.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """
        Get method defaults for the trainer.

        Returns:
            Method info.
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self, cameras: Cameras, progress_callback: Optional[ProgressCallback] = None) -> Iterable[RenderOutput]:  # [h w c]
        """
        Render images.

        Args:
            cameras: Cameras.
            progress_callback: Callback for progress.
        """
        raise NotImplementedError()

    @abstractmethod
    def train_iteration(self, step: int):
        """
        Train one iteration.

        Args:
            step: Current step.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: Path):
        """
        Save model.

        Args:
            path: Path to save.
        """
        raise NotImplementedError()


class RayMethod(Method):
    name: str

    def __init__(self, batch_size, seed: int = 42, config_overrides: Optional[Dict[str, Any]] = None, xnp=np):
        self.batch_size = batch_size
        self.train_dataset: Optional[Dataset] = None
        self.train_images = None
        self.num_iterations: Optional[int] = None
        self.xnp = xnp
        self.config_overrides = {}
        self.config_overrides.update(config_overrides or {})
        self._rng: np.random.Generator = xnp.random.default_rng(seed)

    def get_name(self):
        return self.name

    @abstractmethod
    def render_rays(self, origins: np.ndarray, directions: np.ndarray, nears_fars: Optional[np.ndarray]) -> RenderOutput:  # batch 3  # batch 3  # batch 3
        """
        Render rays.

        Args:
            origins: Ray origins.
            directions: Ray directions.
            nears_fars: Near and far planes.
        """
        raise NotImplementedError()

    @abstractmethod
    def train_iteration_rays(self, step: int, origins: np.ndarray, directions: np.ndarray, nears_fars: Optional[np.ndarray], colors: np.ndarray):  # [batch 3]  # [batch 3]  # [batch 2]  # [batch c]
        """
        Train one iteration.

        Args:
            step: Current step.
            origins: Ray origins.
            directions: Ray directions.
            nears_fars: Near and far planes.
            colors: Colors.
        """
        raise NotImplementedError()

    def setup_train(self, train_dataset: Dataset, *, num_iterations: Optional[int] = None, config_overrides: Optional[Dict[str, Any]] = None):
        self.train_dataset = train_dataset
        train_images, self.train_dataset.images = train_dataset.images, None
        assert train_images is not None, "train_dataset must have images loaded. Use `load_features` to load them."
        self.train_images = padded_stack(train_images)
        train_dataset.images = None  # Free memory
        self.num_iterations = num_iterations
        self.config_overrides.update(config_overrides or {})

    def train_iteration(self, step: int):
        assert self.train_dataset is not None, "setup_train must be called before train_iteration"
        assert self.train_images is not None, "train_dataset must have images"
        assert self.train_dataset.cameras.image_sizes is not None, "train_dataset must have image_sizes"
        xnp = self.xnp
        camera_indices = self._rng.integers(0, len(self.train_dataset.cameras), (self.batch_size,), dtype=xnp.int32)
        wh = self.train_dataset.cameras.image_sizes[camera_indices]
        x = xnp.random.randint(0, wh[..., 0])
        y = xnp.random.randint(0, wh[..., 1])
        xy = xnp.stack([x, y], -1)
        cameras = self.train_dataset.cameras[camera_indices]
        origins, directions = cameras.get_rays(xy, xnp=cast(Any, xnp))
        colors = self.train_images[camera_indices, xy[..., 1], xy[..., 0]]
        return self.train_iteration_rays(step=step, origins=origins, directions=directions, nears_fars=cameras.nears_fars, colors=colors)

    def render(self, cameras: Cameras, progress_callback: Optional[ProgressCallback] = None) -> Iterable[RenderOutput]:
        assert cameras.image_sizes is not None, "cameras must have image_sizes"
        xnp = self.xnp
        batch_size = self.batch_size
        sizes = cameras.image_sizes
        total_batches = ((sizes.prod(-1) + batch_size - 1) // batch_size).sum().item()
        global_i = 0
        if progress_callback:
            progress_callback(CurrentProgress(i=global_i, total=total_batches, image_i=0, image_total=len(sizes)))
        for i, image_size in enumerate(sizes.tolist()):
            w, h = image_size
            local_cameras = cameras[i : i + 1, None]
            xy = local_cameras.get_image_pixels(image_size)
            outputs: List[RenderOutput] = []
            for xy in batched(xy, batch_size):
                origins, directions = local_cameras.get_rays(xy[None], xnp=cast(Any, xnp))
                _outputs = self.render_rays(origins=origins[0], directions=directions[0], nears_fars=local_cameras[0].nears_fars)
                outputs.append(_outputs)
                global_i += 1
                if progress_callback:
                    progress_callback(CurrentProgress(i=global_i, total=total_batches, image_i=i, image_total=len(sizes)))
            # The following is not supported by mypy yet.
            yield {  # type: ignore
                k: np.concatenate([x[k] for x in outputs], 0).reshape((h, w, -1)) for k in outputs[0].keys()  # type: ignore
            }


@runtime_checkable
@generate_interface
class EvaluationProtocol(Protocol):
    def get_name(self) -> str:
        ...
        
    def render(self, method: Method, dataset: Dataset, progress_callback: Optional[ProgressCallback] = None) -> Iterable[RenderOutput]:
        ...

    def evaluate(self, predictions: Iterable[RenderOutput], dataset: Dataset) -> Iterable[Dict[str, Union[float, int]]]:
        ...

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        ...
