from abc import abstractmethod
from typing import Optional, Iterable, List, Dict, Any, cast, Union, Sequence, TypeVar, TYPE_CHECKING
from dataclasses import dataclass
import dataclasses
import os
import numpy as np
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
try:
    from typing import Generic
except ImportError:
    from typing_extensions import Generic
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
try:
    from typing import runtime_checkable
except ImportError:
    from typing_extensions import runtime_checkable
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
try:
    from typing import get_args as get_args
    from typing import get_origin as get_origin
except ImportError:
    from typing_extensions import get_args as get_args
    from typing_extensions import get_origin as get_origin
try:
    from typing import NotRequired
    from typing import Required
    from typing import TypedDict
except ImportError:
    from typing_extensions import NotRequired
    from typing_extensions import Required
    from typing_extensions import TypedDict
try:
    from typing import FrozenSet
except ImportError:
    from typing_extensions import FrozenSet
from .utils import padded_stack, generate_interface, TTensor
from .utils import _xnp_copy, _get_xnp, _xnp_astype


NB_PREFIX = os.path.expanduser(os.environ.get("NERFBASELINES_PREFIX", "~/.cache/nerfbaselines"))
ColorSpace = Literal["srgb", "linear"]
CameraModel = Literal["pinhole", "opencv", "opencv_fisheye", "full_opencv"]
DatasetFeature = Literal["color", "points3D_xyz", "points3D_rgb"]


def camera_model_to_int(camera_model: CameraModel) -> int:
    camera_models = get_args(CameraModel)
    if camera_model not in camera_models:
        raise ValueError(f"Unknown camera model {camera_model}, known models are {camera_models}")
    return get_args(CameraModel).index(camera_model)


def camera_model_from_int(i: int) -> CameraModel:
    camera_models = get_args(CameraModel)
    if i >= len(camera_models):
        raise ValueError(f"Unknown camera model with index {i}, known models are {camera_models}")
    return get_args(CameraModel)[i]


@dataclass(frozen=True)
class GenericCameras(Generic[TTensor]):
    poses: TTensor  # [N, (R, t)]
    intrinsics: TTensor  # [N, (fx,fy,cx,cy)]

    camera_types: TTensor  # [N]
    distortion_parameters: TTensor  # [N, num_params]
    image_sizes: TTensor  # [N, 2]

    nears_fars: Optional[TTensor]  # [N, 2]
    metadata: Optional[TTensor] = None


    def __len__(self):
        return 1 if len(self.poses.shape) == 2 else len(self.poses)

    def item(self):
        assert len(self) == 1, "Cameras must have exactly one element to be converted to a single camera"
        return self if len(self.poses.shape) == 2 else self[0]

    def __getitem__(self, index):
        return type(self)(
            poses=self.poses[index],
            intrinsics=self.intrinsics[index],
            camera_types=self.camera_types[index],
            distortion_parameters=self.distortion_parameters[index],
            image_sizes=self.image_sizes[index],
            nears_fars=self.nears_fars[index] if self.nears_fars is not None else None,
            metadata=self.metadata[index] if self.metadata is not None else None,
        )

    def __setitem__(self, index, value):
        assert (self.image_sizes is None) == (value.image_sizes is None), "Either both or none of the cameras must have image sizes"
        assert (self.nears_fars is None) == (value.nears_fars is None), "Either both or none of the cameras must have nears and fars"
        self.poses[index] = value.poses
        self.intrinsics[index] = value.intrinsics
        self.camera_types[index] = value.camera_types
        self.distortion_parameters[index] = value.distortion_parameters
        self.image_sizes[index] = value.image_sizes
        if self.nears_fars is not None:
            self.nears_fars[index] = value.nears_fars
        if self.metadata is not None:
            self.metadata[index] = value.metadata

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def cat(cls, values: Sequence[Self]) -> Self:
        xnp = _get_xnp(values[0].poses)
        nears_fars = None
        metadata = None
        if any(v.nears_fars is not None for v in values):
            assert all(v.nears_fars is not None for v in values), "Either all or none of the cameras must have nears and fars"
            nears_fars = xnp.concatenate([cast(TTensor, v.nears_fars) for v in values])
        if any(v.metadata is not None for v in values):
            assert all(v.metadata is not None for v in values), "Either all or none of the cameras must have metadata"
            metadata = xnp.concatenate([cast(TTensor, v.metadata) for v in values])
        return cls(
            poses=xnp.concatenate([v.poses for v in values]),
            intrinsics=xnp.concatenate([v.intrinsics for v in values]),
            camera_types=xnp.concatenate([v.camera_types for v in values]),
            distortion_parameters=xnp.concatenate([v.distortion_parameters for v in values]),
            image_sizes=xnp.concatenate([cast(TTensor, v.image_sizes) for v in values]),
            nears_fars=nears_fars,
            metadata=metadata,
        )

    def replace(self, **changes) -> Self:
        return dataclasses.replace(self, **changes)

    def with_image_sizes(self, image_sizes: TTensor) -> Self:
        xnp = _get_xnp(self.poses)
        multipliers = _xnp_astype(image_sizes, self.intrinsics.dtype, xnp=xnp) / self.image_sizes
        multipliers = xnp.concatenate([multipliers, multipliers], -1)
        intrinsics = self.intrinsics * multipliers
        return self.replace( image_sizes=image_sizes, intrinsics=intrinsics)

    def with_metadata(self, metadata: TTensor) -> Self:
        return self.replace(metadata=metadata)

    def clone(self) -> Self:
        xnp = _get_xnp(self.poses)
        return self.replace(
            poses=_xnp_copy(self.poses, xnp),
            intrinsics=_xnp_copy(self.intrinsics, xnp),
            camera_types=_xnp_copy(self.camera_types, xnp),
            distortion_parameters=_xnp_copy(self.distortion_parameters, xnp),
            image_sizes=_xnp_copy(self.image_sizes, xnp) if self.image_sizes is not None else None,
            nears_fars=_xnp_copy(self.nears_fars, xnp) if self.nears_fars is not None else None,
            metadata=_xnp_copy(self.metadata, xnp) if self.metadata is not None else None)


class Cameras(GenericCameras[np.ndarray]):
    pass


class _IncompleteDataset(TypedDict, total=True):
    cameras: Cameras  # [N]

    file_paths: List[str]
    sampling_mask_paths: Optional[List[str]]
    file_paths_root: Optional[str]
    metadata: Dict
    sampling_masks: Optional[Union[np.ndarray, List[np.ndarray]]]  # [N][H, W]
    points3D_xyz: Optional[np.ndarray]  # [M, 3]
    points3D_rgb: Optional[np.ndarray]  # [M, 3]


class UnloadedDataset(_IncompleteDataset):
    images: NotRequired[Optional[Union[np.ndarray, List[np.ndarray]]]]  # [N][H, W, 3]


class Dataset(_IncompleteDataset):
    images: Union[np.ndarray, List[np.ndarray]]  # [N][H, W, 3]


class RenderOutput(TypedDict, total=False):
    color: Required[np.ndarray]  # [h w 3]
    depth: np.ndarray  # [h w]
    accumulation: np.ndarray  # [h w]


class OptimizeEmbeddingsOutput(TypedDict):
    embedding: np.ndarray
    render_output: RenderOutput
    metrics: NotRequired[Dict[str, Sequence[float]]]


class MethodInfo(TypedDict, total=False):
    name: Required[str]
    required_features: FrozenSet[DatasetFeature]
    supported_camera_models: FrozenSet


class ModelInfo(TypedDict, total=False):
    name: Required[str]
    num_iterations: Required[int]
    loaded_step: Optional[int]
    loaded_checkpoint: Optional[str]
    batch_size: int
    eval_batch_size: int
    required_features: FrozenSet[DatasetFeature]
    supported_camera_models: FrozenSet
    hparams: Dict[str, Any]


@runtime_checkable
@generate_interface
class Method(Protocol):
    def __init__(self, 
                 *,
                 checkpoint: Union[str, None] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[Dict[str, Any]] = None):
        pass

    @classmethod
    def install(cls):
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
    def optimize_embeddings(
        self, 
        dataset: Dataset,
        embeddings: Optional[np.ndarray] = None
    ) -> Iterable[OptimizeEmbeddingsOutput]:
        """
        Optimize embeddings for each image in the dataset.

        Args:
            dataset: Dataset.
            embeddings: Optional initial embeddings.
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self, cameras: Cameras, embeddings: Optional[np.ndarray] = None) -> Iterable[RenderOutput]:  # [h w c]
        """
        Render images.

        Args:
            cameras: Cameras.
            embeddings: Optional image embeddings.
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
    def save(self, path: str):
        """
        Save model.

        Args:
            path: Path to save.
        """
        raise NotImplementedError()


def _batched(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i : i + batch_size]


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
        self.train_cameras = None
        self._rng: np.random.Generator = xnp.random.default_rng(seed)

    def get_name(self):
        return self.name

    @abstractmethod
    def render_rays(self, origins: np.ndarray, directions: np.ndarray, nears_fars: Optional[np.ndarray], embeddings: Optional[np.ndarray] = None) -> RenderOutput:  # batch 3  # batch 3  # batch 3
        """
        Render rays.

        Args:
            origins: Ray origins.
            directions: Ray directions.
            nears_fars: Near and far planes.
            embeddings: Optional image embeddings.
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

    def optimize_embeddings(self, dataset: Dataset, embeddings: Optional[np.ndarray] = None) -> Iterable[OptimizeEmbeddingsOutput]:
        raise NotImplementedError()

    def setup_train(self, train_dataset: Dataset, *, num_iterations: Optional[int] = None, config_overrides: Optional[Dict[str, Any]] = None):
        self.train_dataset = train_dataset
        # Free memory
        train_images, self.train_dataset["images"] = train_dataset["images"], None  # type: ignore
        assert train_images is not None, "train_dataset must have images loaded. Use `load_features` to load them."
        self.train_images = padded_stack(train_images)
        self.num_iterations = num_iterations
        self.train_cameras = train_dataset["cameras"]
        self.config_overrides.update(config_overrides or {})

    def train_iteration(self, step: int):
        from . import cameras as _cameras
        assert self.train_dataset is not None, "setup_train must be called before train_iteration"
        assert self.train_images is not None, "train_dataset must have images"
        assert self.train_cameras is not None, "setup_train must be called before train_iteration"
        xnp = self.xnp
        camera_indices = self._rng.integers(0, len(self.train_cameras), (self.batch_size,), dtype=xnp.int32)
        wh = self.train_cameras.image_sizes[camera_indices]
        x = xnp.random.randint(0, wh[..., 0])
        y = xnp.random.randint(0, wh[..., 1])
        xy = xnp.stack([x, y], -1)
        cameras = self.train_cameras[camera_indices]
        origins, directions = _cameras.get_rays(cameras, xy)
        colors = self.train_images[camera_indices, xy[..., 1], xy[..., 0]]
        return self.train_iteration_rays(step=step, origins=origins, directions=directions, nears_fars=cameras.nears_fars, colors=colors)

    def render(self, cameras: Cameras, embeddings: Optional[np.ndarray] = None) -> Iterable[RenderOutput]:
        from . import cameras as _cameras
        assert cameras.image_sizes is not None, "cameras must have image_sizes"
        batch_size = self.batch_size
        sizes = cameras.image_sizes
        global_i = 0
        for i, image_size in enumerate(sizes.tolist()):
            w, h = image_size
            local_cameras = cameras[i : i + 1, None]
            xy = _cameras.get_image_pixels(image_size)
            outputs: List[RenderOutput] = []
            for xy in _batched(xy, batch_size):
                origins, directions = _cameras.get_rays(local_cameras, xy[None])
                local_embedding = embeddings[i:i+1] if embeddings is not None else None
                _outputs = self.render_rays(origins=origins[0], directions=directions[0], nears_fars=local_cameras[0].nears_fars, embeddings=local_embedding)
                outputs.append(_outputs)
                global_i += 1
            # The following is not supported by mypy yet.
            yield {  # type: ignore
                k: np.concatenate([x[k] for x in outputs], 0).reshape((h, w, -1)) for k in outputs[0].keys()  # type: ignore
            }


@runtime_checkable
@generate_interface
class EvaluationProtocol(Protocol):
    def get_name(self) -> str:
        ...
        
    def render(self, method: Method, dataset: Dataset) -> Iterable[RenderOutput]:
        ...

    def evaluate(self, predictions: Iterable[RenderOutput], dataset: Dataset) -> Iterable[Dict[str, Union[float, int]]]:
        ...

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        ...
