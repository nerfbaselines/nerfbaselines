from abc import abstractmethod
from typing import Tuple, Optional, Callable, Iterable, List
from dataclasses import dataclass, field
import dataclasses
import os
from pathlib import Path
from abc import abstractmethod
import numpy as np
try:
    from typing import Protocol, Literal
except ImportError:
    from typing_extensions import Protocol, Literal
try:
    from typing import FrozenSet
except ImportError:
    from typing_extensions import FrozenSet
from .distortion import Distortions
from .utils import get_rays


ColorSpace = Literal["srgb", "linear"]


@dataclass
class Dataset:
    camera_poses: np.ndarray  # [N, (R, t)]
    camera_intrinsics_normalized: np.ndarray  # [N, (nfx,nfy,ncx,ncy)]
    camera_distortions: Distortions  # [N, (type, params)]
    # camera_ids: Tensor
    image_sizes: Optional[np.ndarray]  # [N, 2]
    nears_fars: np.ndarray  # [N, 2]

    file_paths: List[str]
    sampling_mask_paths: Optional[List[str]] = None
    file_paths_root: Optional[Path] = None

    images: Optional[np.array] = None  # [N, H, W, 3]
    sampling_masks: Optional[np.array] = None  # [N, H, W]
    points3D_xyz: Optional[np.ndarray] = None  # [M, 3]
    points3D_rgb: Optional[np.ndarray] = None  # [M, 3]

    metadata: Optional[dict] = field(default_factory=dict)
    color_space: Optional[ColorSpace] = None

    @property
    def camera_intrinsics(self):
        assert self.image_sizes is not None
        return self.camera_intrinsics_normalized * self.image_sizes[..., :1]

    def __post_init__(self):
        if self.file_paths_root is None:
            self.file_paths_root = Path(os.path.commonpath(self.file_paths))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, i) -> 'Dataset':
        assert isinstance(i, slice) or isinstance(i, int) or isinstance(i, np.ndarray)
        def index(obj):
            if obj is None:
                return None
            if isinstance(obj, Distortions):
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
        return dataclasses.replace(self, **{
            k: index(v) for k, v in self.__dict__.items()
            if k not in {"file_paths_root", "points3D_xyz", "points3D_rgb", "metadata", "color_space"}
        })


@dataclass(frozen=True)
class CurrentProgress:
    i: int
    total: int
    image_i: int
    image_total: int

ProgressCallback = Callable[[CurrentProgress], None]

def batched(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i:i+batch_size]


DatasetFeature = Literal["color", "points3D_xyz"]


@dataclass
class MethodInfo:
    loaded_step: Optional[int] = None
    num_iterations: Optional[int] = None
    batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None
    required_features: FrozenSet[DatasetFeature] = field(default_factory=frozenset)
    supports_undistortion: bool = False


class Method(Protocol):
    @property
    @abstractmethod
    def info(self) -> MethodInfo:
        """
        Get method defaults for the trainer.

        Returns:
            Method info.
        """
        return MethodInfo()

    @abstractmethod
    def render(self,
               poses: np.ndarray, # batch 3 4,
               intrinsics: np.ndarray, # [batch 3 4],
               sizes: np.ndarray,
               nears_fars: np.ndarray, # [batch 2"],
               distortions: Optional[Distortions] = None,
               progress_callback: Optional[ProgressCallback] = None) -> Iterable[np.ndarray]:  # [h w c]
        """
        Render images.

        Args:
            poses: Camera poses.
            intrinsics: Camera intrinsics.
            sizes: Image sizes.
            nears_fars: Near and far planes.
            distortions: Distortions.
            progress_callback: Callback for progress.
        """
        raise NotImplementedError()

    @abstractmethod
    def setup_train(self,
                    train_dataset: Dataset, *,
                    num_iterations: int):
        """
        Setup training data, model, optimizer, etc.

        Args:
            train_dataset: Training dataset.
            num_iterations: Number of iterations to train.
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


class RayMethod(Method):
    def __init__(self, batch_size, seed: int = 42):
        self.batch_size = batch_size
        self.train_dataset = None
        self.num_iterations = None
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def render_rays(self,
                    origins: np.ndarray, # batch 3
                    directions: np.ndarray, # batch 3
                    nears_fars: np.ndarray): # batch 3
        """
        Render rays.

        Args:
            origins: Ray origins.
            directions: Ray directions.
            nears_fars: Near and far planes.
        """
        raise NotImplementedError()

    @abstractmethod
    def train_iteration_rays(self,
                             step: int,
                             origins: np.ndarray, # [batch 3]
                             directions: np.ndarray, # [batch 3]
                             nears_fars: np.ndarray, # [batch 2]
                             colors: np.ndarray): # [batch c]
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

    def setup_train(self, train_dataset: Dataset, *, num_iterations: int):
        self.train_dataset = Dataset
        self.num_iterations = num_iterations

    def train_iteration(self, step: int):
        camera_indices = self._rng.randint(0, len(self.train_dataset.images), (self.batch_size,), dtype=np.int32)
        wh = self.train_dataset.image_sizes[camera_indices]
        areas = wh.prod(-1)
        local_indices = np.clip(np.floor(self._rng.rand((self.batch_size,)) * areas), None, areas).astype(np.int32)
        xy = np.stack([local_indices % wh[..., 0], local_indices // wh[..., 0]], -1)
        nears_fars = self.train_dataset.nears_fars[camera_indices]
        origins, directions = get_rays(self.train_dataset.camera_poses[camera_indices],
                                       self.train_dataset.camera_intrinsics[camera_indices],
                                       xy,
                                       self.train_dataset.camera_distortions[camera_indices] if self.train_dataset.camera_distortions is not None else None)
        colors = self.train_dataset.images[camera_indices, xy[..., 1], xy[..., 0]]
        return self.train_iteration_rays(step=step, origins=origins, directions=directions, nears_fars=nears_fars, colors=colors)

    def render(self,
               poses: np.ndarray, # ["batch 3 4"],
               intrinsics: np.ndarray, # Float32, "batch 3 4",
               sizes: np.ndarray, # Int32, "batch 2",
               nears_fars: np.ndarray, # Float32, "batch 2"],
               distortions: Optional[Distortions] = None,
               progress_callback: Optional[ProgressCallback] = None) -> Iterable[np.ndarray]:
        batch_size = self.batch_size
        assert len(sizes) == len(poses) == len(intrinsics)
        assert len(sizes.shape) == 2
        total_batches = ((sizes.prod(-1) + batch_size - 1) // batch_size).sum().item()
        global_i = 0
        if progress_callback:
            progress_callback(CurrentProgress(i=global_i, total=total_batches, image_i=0, image_total=len(sizes)))
        for i, image_size in enumerate(sizes.long().detach().cpu().tolist()):
            w, h = image_size
            xy = np.stack(np.meshgrid(np.arange(w), np.arange(h), indexing="xy"), -1).reshape(-1, 2)
            outputs = []
            for xy in batched(xy, batch_size):
                origins, directions = get_rays(poses[i:i+1],
                                               intrinsics[i:i+1],
                                               xy,
                                               distortions[i:i+1] if distortions is not None else None)
                _outputs = self.render_rays(origins=origins,
                                            directions=directions,
                                            nears_fars=nears_fars[i:i+1])
                outputs.append(_outputs)
                global_i += 1
                if progress_callback:
                    progress_callback(CurrentProgress(i=global_i, total=total_batches, image_i=i, image_total=len(sizes)))
            yield {
                np.concatenate([x[k] for x in outputs], 0).reshape((h, w, -1))
                for k in outputs[0].keys()
            }