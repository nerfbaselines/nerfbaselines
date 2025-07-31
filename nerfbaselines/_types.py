import sys
from abc import abstractmethod
import typing
from typing import Optional, Iterable, List, Dict, Any, cast, Union, Sequence, TYPE_CHECKING, overload, TypeVar, Iterator, Callable, Tuple
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

if TYPE_CHECKING:
    from .backends import CondaBackendSpec, DockerBackendSpec, ApptainerBackendSpec
else:
    CondaBackendSpec = Any
    DockerBackendSpec = Any
    ApptainerBackendSpec = Any

if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp


NB_PREFIX = os.path.expanduser(os.environ.get("NERFBASELINES_PREFIX", "~/.cache/nerfbaselines"))
ColorSpace = Literal["srgb", "linear"]
CameraModel = Literal["pinhole", "opencv", "opencv_fisheye", "full_opencv"]
BackendName = Literal["conda", "docker", "apptainer", "python"]
DatasetFeature = Literal["color", "points3D_xyz", "points3D_rgb", 
                         "images_points3D_indices", "images_points2D_xy",
                         "points3D_error"]
TTensor = TypeVar("TTensor", np.ndarray, "torch.Tensor", "jnp.ndarray")
TTensor_co = TypeVar("TTensor_co", np.ndarray, "torch.Tensor", "jnp.ndarray", covariant=True)


@overload
def _get_xnp(tensor: np.ndarray):
    return np

@overload
def _get_xnp(tensor: 'jnp.ndarray'):
    return cast('jnp', sys.modules["jax.numpy"])

@overload
def _get_xnp(tensor: 'torch.Tensor'):
    return cast('torch', sys.modules["torch"])


def _get_xnp(tensor: TTensor):
    if isinstance(tensor, np.ndarray):
        return np
    if tensor.__module__.startswith("jax"):
        return cast('jnp', sys.modules["jax.numpy"])
    if tensor.__module__ == "torch":
        return cast('torch', sys.modules["torch"])
    raise ValueError(f"Unknown tensor type {type(tensor)}")


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


class GenericCameras(Protocol[TTensor_co]):
    @property
    def poses(self) -> TTensor_co:
        """Camera-to-world matrices, [N, (R, t)]"""
        ...

    @property
    def intrinsics(self) -> TTensor_co:
        """Intrinsics, [N, (fx,fy,cx,cy)]"""
        ...

    @property
    def camera_models(self) -> TTensor_co:
        """Camera models, [N]"""
        ...

    @property
    def distortion_parameters(self) -> TTensor_co:
        """Distortion parameters, [N, num_params]"""
        ...

    @property
    def image_sizes(self) -> TTensor_co:
        """Image sizes, [N, 2]"""
        ...

    @property
    def nears_fars(self) -> Optional[TTensor_co]:
        """Near and far planes, [N, 2]"""
        ...

    @property
    def metadata(self) -> Optional[TTensor_co]:
        """Metadata, [N, ...]"""
        ...

    def __len__(self) -> int:
        ...

    def item(self) -> Self:
        """Returns a single camera if there is only one. Otherwise raises an error."""
        ...

    def __getitem__(self, index) -> Self:
        ...

    def __setitem__(self, index, value: Self) -> None:
        ...

    def __iter__(self) -> Iterator[Self]:
        ...

    @classmethod
    def cat(cls, values: Sequence[Self]) -> Self:
        ...

    def replace(self, **changes) -> Self:
        ...

    def apply(self, fn: Callable[[TTensor_co, str], TTensor]) -> 'GenericCameras[TTensor]':
        ...


@runtime_checkable
class Cameras(GenericCameras[np.ndarray], Protocol):
    pass


@dataclass(frozen=True)
class GenericCamerasImpl(Generic[TTensor_co]):
    poses: TTensor_co  # [N, (R, t)]
    intrinsics: TTensor_co  # [N, (fx,fy,cx,cy)]

    camera_models: TTensor_co  # [N]
    distortion_parameters: TTensor_co  # [N, num_params]
    image_sizes: TTensor_co  # [N, 2]

    nears_fars: Optional[TTensor_co]  # [N, 2]
    metadata: Optional[TTensor_co] = None

    def __len__(self) -> int:
        return 1 if len(self.poses.shape) == 2 else len(self.poses)

    def item(self):
        assert len(self) == 1, "Cameras must have exactly one element to be converted to a single camera"
        return self if len(self.poses.shape) == 2 else self[0]

    def __getitem__(self, index):
        return type(self)(
            poses=self.poses[index],
            intrinsics=self.intrinsics[index],
            camera_models=self.camera_models[index],
            distortion_parameters=self.distortion_parameters[index],
            image_sizes=self.image_sizes[index],
            nears_fars=self.nears_fars[index] if self.nears_fars is not None else None,
            metadata=self.metadata[index] if self.metadata is not None else None,
        )

    def __setitem__(self, index, value: Self) -> None:
        assert (self.image_sizes is None) == (value.image_sizes is None), "Either both or none of the cameras must have image sizes"
        assert (self.nears_fars is None) == (value.nears_fars is None), "Either both or none of the cameras must have nears and fars"
        self.poses[index] = value.poses
        self.intrinsics[index] = value.intrinsics
        self.camera_models[index] = value.camera_models
        self.distortion_parameters[index] = value.distortion_parameters
        self.image_sizes[index] = value.image_sizes
        if self.nears_fars is not None:
            self.nears_fars[index] = cast(TTensor_co, value.nears_fars)
        if self.metadata is not None:
            self.metadata[index] = cast(TTensor_co, value.metadata)

    def __iter__(self) -> Iterator[Self]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def cat(cls, values: Sequence[Self]) -> Self:
        xnp = _get_xnp(values[0].poses)
        nears_fars: Optional[TTensor_co] = None
        metadata: Optional[TTensor_co] = None
        if any(v.nears_fars is not None for v in values):
            assert all(v.nears_fars is not None for v in values), "Either all or none of the cameras must have nears and fars"
            nears_fars = xnp.concatenate([cast(TTensor_co, v.nears_fars) for v in values])
        if any(v.metadata is not None for v in values):
            assert all(v.metadata is not None for v in values), "Either all or none of the cameras must have metadata"
            metadata = xnp.concatenate([cast(TTensor_co, v.metadata) for v in values])
        return cls(
            poses=xnp.concatenate([v.poses for v in values]),
            intrinsics=xnp.concatenate([v.intrinsics for v in values]),
            camera_models=xnp.concatenate([v.camera_models for v in values]),
            distortion_parameters=xnp.concatenate([v.distortion_parameters for v in values]),
            image_sizes=xnp.concatenate([cast(TTensor_co, v.image_sizes) for v in values]),
            nears_fars=nears_fars,
            metadata=metadata,
        )

    def replace(self, **changes) -> Self:
        return dataclasses.replace(self, **changes)

    def apply(self, fn: Callable[[TTensor_co, str], TTensor]) -> 'GenericCamerasImpl[TTensor]':
        return GenericCamerasImpl[TTensor](
            poses=fn(self.poses, "poses"),
            intrinsics=fn(self.intrinsics, "intrinsics"),
            camera_models=fn(self.camera_models, "camera_models"),
            distortion_parameters=fn(self.distortion_parameters, "distortion_parameters"),
            image_sizes=fn(self.image_sizes, "image_sizes"),
            nears_fars=fn(cast(TTensor_co, self.nears_fars), "nears_fars") if self.nears_fars is not None else None,
            metadata=fn(cast(TTensor_co, self.metadata), "metadata") if self.metadata is not None else None,
        )


def new_cameras(
    *,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    camera_models: np.ndarray,
    image_sizes: np.ndarray,
    distortion_parameters: Optional[np.ndarray] = None,
    nears_fars: Optional[np.ndarray] = None,
    metadata: Optional[np.ndarray] = None,
) -> Cameras:
    if distortion_parameters is None:
        shape = list(intrinsics.shape)
        shape[-1] = 0
        distortion_parameters = np.zeros(tuple(shape), dtype=intrinsics.dtype)
    return GenericCamerasImpl[np.ndarray](
        poses=poses,
        intrinsics=intrinsics,
        camera_models=camera_models,
        distortion_parameters=distortion_parameters,
        image_sizes=image_sizes,
        nears_fars=nears_fars,
        metadata=metadata)
    

class _IncompleteDataset(TypedDict, total=True):
    cameras: Cameras  # [N]

    image_paths: List[str]
    image_paths_root: str
    mask_paths: Optional[List[str]]
    mask_paths_root: Optional[str]
    metadata: Dict
    masks: Optional[Union[np.ndarray, List[np.ndarray]]]  # [N][H, W]
    points3D_xyz: Optional[np.ndarray]  # [M, 3]
    points3D_rgb: Optional[np.ndarray]  # [M, 3]
    points3D_error: Optional[np.ndarray]  # [M]
    images_points3D_indices: Optional[List[np.ndarray]]  # [N][<M]
    images_points2D_xy: Optional[List[np.ndarray]]  # [N][<M, 2]


class UnloadedDataset(_IncompleteDataset):
    images: NotRequired[Optional[Union[np.ndarray, List[np.ndarray]]]]  # [N][H, W, 3]


class Dataset(_IncompleteDataset):
    images: Union[np.ndarray, List[np.ndarray]]  # [N][H, W, 3]


@overload
def new_dataset(*,
                cameras: Cameras,
                image_paths: Sequence[str],
                image_paths_root: Optional[str] = ...,
                images: Union[np.ndarray, List[np.ndarray]],
                mask_paths: Optional[Sequence[str]] = ...,
                mask_paths_root: Optional[str] = ...,
                masks: Optional[Union[np.ndarray, List[np.ndarray]]] = ...,  # [N][H, W]
                points3D_xyz: Optional[np.ndarray] = ...,  # [M, 3]
                points3D_rgb: Optional[np.ndarray] = ...,  # [M, 3]
                points3D_error: Optional[np.ndarray] = ...,  # [M]
                images_points3D_indices: Optional[Sequence[np.ndarray]] = ...,  # [N][<M]
                images_points2D_xy: Optional[Sequence[np.ndarray]] = ...,  # [N][<M, 2]
                metadata: Optional[Dict] = ...) -> Dataset:
    ...


@overload
def new_dataset(*,
                cameras: Cameras,
                image_paths: Sequence[str],
                image_paths_root: Optional[str] = ...,
                images: Literal[None] = None,
                mask_paths: Optional[Sequence[str]] = ...,
                mask_paths_root: Optional[str] = ...,
                masks: Optional[Union[np.ndarray, List[np.ndarray]]] = ...,  # [N][H, W]
                points3D_xyz: Optional[np.ndarray] = ...,  # [M, 3]
                points3D_rgb: Optional[np.ndarray] = ...,  # [M, 3]
                points3D_error: Optional[np.ndarray] = ...,  # [M]
                images_points3D_indices: Optional[Sequence[np.ndarray]] = ...,  # [N][<M]
                images_points2D_xy: Optional[Sequence[np.ndarray]] = ...,  # [N][<M, 2]
                metadata: Optional[Dict] = ...) -> UnloadedDataset:
    ...


def new_dataset(*,
                cameras: Cameras,
                image_paths: Sequence[str],
                image_paths_root: Optional[str] = None,
                images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,  # [N][H, W, 3]
                mask_paths: Optional[Sequence[str]] = None,
                mask_paths_root: Optional[str] = None,
                masks: Optional[Union[np.ndarray, List[np.ndarray]]] = None,  # [N][H, W]
                points3D_xyz: Optional[np.ndarray] = None,  # [M, 3]
                points3D_rgb: Optional[np.ndarray] = None,  # [M, 3]
                points3D_error: Optional[np.ndarray] = None,  # [M]
                images_points3D_indices: Optional[Sequence[np.ndarray]] = None,  # [N][<M]
                images_points2D_xy: Optional[Sequence[np.ndarray]] = None,  # [N][<M, 2]
                metadata: Optional[Dict] = None) -> Union[UnloadedDataset, Dataset]:
    if image_paths_root is None:
        image_paths_root = os.path.commonpath(image_paths)
    if mask_paths_root is None and mask_paths is not None:
        mask_paths_root = os.path.commonpath(mask_paths)
    if image_paths_root is None:
        image_paths_root = os.path.commonpath(image_paths)
    if metadata is None:
        metadata = {}
    return UnloadedDataset(
        cameras=cameras,
        image_paths=list(image_paths),
        mask_paths=list(mask_paths) if mask_paths is not None else None,
        mask_paths_root=mask_paths_root,
        image_paths_root=image_paths_root,
        images=images,
        masks=masks,
        points3D_xyz=points3D_xyz,
        points3D_rgb=points3D_rgb,
        points3D_error=points3D_error,
        images_points3D_indices=list(images_points3D_indices) if images_points3D_indices is not None else None,
        images_points2D_xy=list(images_points2D_xy) if images_points2D_xy is not None else None,
        metadata=metadata
    )


RenderOutput = Dict[str, np.ndarray]
# NOTE: Type intersection is not supported for now
# color: Required[np.ndarray]  # [h w 3]
# depth: np.ndarray  # [h w]
# accumulation: np.ndarray  # [h w]


class OptimizeEmbeddingOutput(TypedDict):
    embedding: Required[np.ndarray]
    render_output: NotRequired[RenderOutput]
    metrics: NotRequired[Dict[str, Sequence[float]]]


class RenderOutputType(TypedDict, total=False):
    name: Required[str]
    type: Literal["color", "depth"]


class MethodInfo(TypedDict, total=False):
    method_id: Required[str]
    required_features: FrozenSet[DatasetFeature]
    supported_camera_models: FrozenSet[CameraModel]
    supported_outputs: Tuple[Union[str, RenderOutputType], ...]
    can_resume_training: bool
    viewer_default_resolution: Union[int, Tuple[int, int]]


class ModelInfo(TypedDict, total=False):
    method_id: Required[str]
    num_iterations: Required[int]
    loaded_step: Optional[int]
    loaded_checkpoint: Optional[str]
    batch_size: int
    eval_batch_size: int
    required_features: FrozenSet[DatasetFeature]
    supported_camera_models: FrozenSet
    hparams: Dict[str, Any]
    supported_outputs: Tuple[Union[str, RenderOutputType], ...]
    can_resume_training: bool
    viewer_default_resolution: Union[int, Tuple[int, int]]


class RenderOptions(TypedDict, total=False):
    embedding: Optional[np.ndarray]
    outputs: Tuple[str, ...]
    output_type_dtypes: Dict[str, str]


@runtime_checkable
class Method(Protocol):
    def __init__(self, 
                 *,
                 checkpoint: Union[str, None] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[Dict[str, Any]] = None):
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

    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        """
        Get the embedding for the given image index.

        Args:
            index: Image index.

        Returns:
            Image embedding.
        """
        return None

    def optimize_embedding(self, 
                           dataset: Dataset, *,
                           embedding: Optional[np.ndarray] = None) -> OptimizeEmbeddingOutput:
        """
        Optimize embedding for a single image (passed as a dataset with a single image).

        Args:
            dataset: A dataset with a single image.
            embeddings: Optional initial embedding.
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self, 
               camera: Cameras, *, 
               options: Optional[RenderOptions] = None) -> RenderOutput:  # [h w c]
        """
        Render single image.

        Args:
            camera: Camera from which the scene is to be rendered.
            options: Optional rendering options.
        """
        raise NotImplementedError()

    @abstractmethod
    def train_iteration(self, step: int) -> Dict[str, float]:
        """
        Train one iteration.

        Args:
            step: Current step.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model.

        Args:
            path: Path to save.
        """
        raise NotImplementedError()


@runtime_checkable
class EvaluationProtocol(Protocol):
    def get_name(self) -> str:
        ...
        
    def render(self, method: Method, dataset: Dataset, *, options: Optional[RenderOptions] = None) -> RenderOutput:
        ...

    def evaluate(self, predictions: RenderOutput, dataset: Dataset) -> Dict[str, Union[float, int]]:
        ...

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        ...


class LicenseSpec(TypedDict, total=False):
    name: Required[str]
    url: str


class DatasetSpecMetadata(TypedDict, total=False):
    id: str
    name: str
    description: str
    paper_title: str
    paper_authors: List[str]
    paper_link: str
    link: str
    metrics: List[str]
    default_metric: str
    scenes: List[Dict[str, str]]
    licenses: List[LicenseSpec]


class LoadDatasetFunction(Protocol):
    def __call__(self, 
                 path: str, 
                 split: str, 
                 *,
                 features: Optional[FrozenSet[DatasetFeature]] = None, 
                 **kwargs) -> UnloadedDataset:
        ...


class DownloadDatasetFunction(Protocol):
    def __call__(self, 
                 path: str, 
                 output: str) -> None:
        ...


class TrajectoryFrameAppearance(TypedDict, total=False):
    embedding: Optional[np.ndarray]
    embedding_train_index: Optional[int]


class TrajectoryFrame(TypedDict, total=True):
    pose: np.ndarray
    intrinsics: np.ndarray
    appearance_weights: NotRequired[np.ndarray]


class Trajectory(TypedDict, total=True):
    camera_model: CameraModel
    image_size: Tuple[int, int]
    frames: List[TrajectoryFrame]
    appearances: NotRequired[List[TrajectoryFrameAppearance]]
    fps: float
    source: Any


@runtime_checkable
class LoggerEvent(Protocol):
    def add_scalar(self, tag: str, value: Union[float, int]) -> None:
        ...

    def add_text(self, tag: str, text: str) -> None:
        ...

    def add_image(self, tag: str, image: np.ndarray, display_name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> None:
        ...

    def add_embedding(self, tag: str, embeddings: np.ndarray, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        ...

    def add_plot(self, tag: str, *data: np.ndarray,
                 axes_labels: Optional[Sequence[str]] = None, 
                 title: Optional[str] = None,
                 **kwargs) -> None:
        ...

    def add_histogram(self, tag: str, values: np.ndarray, *, num_bins: Optional[int] = None) -> None:
        ...


@runtime_checkable
class Logger(Protocol):
    def add_event(self, step: int) -> typing.ContextManager[LoggerEvent]:
        ...

    def add_scalar(self, tag: str, value: Union[float, int], step: int) -> None:
        ...

    def add_text(self, tag: str, text: str, step: int) -> None:
        ...

    def add_image(self, tag: str, image: np.ndarray, step: int, *, display_name: Optional[str] = None, description: Optional[str] = None) -> None:
        ...

    def add_embedding(self, tag: str, embeddings: np.ndarray, step: int, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        ...




class OutputArtifact(TypedDict, total=False):
    link: Required[str]
    

# The following will be allowed in Python 3.13
# MethodSpecPresetApplyCondition = TypedDict("MethodSpecPresetApplyCondition", {
#     "dataset": str,
#     "scene": str,
# }, total=False)
# MethodSpecPreset = TypedDict("MethodSpecPreset", {
#     "@apply": List[MethodSpecPresetApplyCondition],
#     "@description": str,
# }, total=False)


ImplementationStatus = Literal["working", "reproducing", "not-working", "working-not-reproducing"]


class MethodSpec(TypedDict, total=False):
    id: Required[str]
    method_class: Required[str]
    conda: NotRequired['CondaBackendSpec']
    docker: NotRequired['DockerBackendSpec']
    apptainer: NotRequired['ApptainerBackendSpec']
    metadata: Dict[str, Any]
    backends_order: List[BackendName]
    presets: Dict[str, Dict[str, Any]]

    required_features: List[DatasetFeature]
    supported_camera_models: List[CameraModel]
    supported_outputs: List[Union[str, RenderOutputType]]
    output_artifacts: Dict[str, OutputArtifact]
    implementation_status: Dict[str, ImplementationStatus]


class LoggerSpec(TypedDict, total=False):
    id: Required[str]
    logger_class: Required[str]


class EvaluationProtocolSpec(TypedDict, total=False):
    id: Required[str]
    evaluation_protocol_class: Required[str]


class DatasetSpec(TypedDict, total=False):
    id: Required[str]
    download_dataset_function: Required[str]
    evaluation_protocol: Union[str, EvaluationProtocolSpec]
    metadata: DatasetSpecMetadata


class DatasetLoaderSpec(TypedDict, total=False):
    id: Required[str]
    load_dataset_function: Required[str]


class DatasetNotFoundError(Exception):
    pass
