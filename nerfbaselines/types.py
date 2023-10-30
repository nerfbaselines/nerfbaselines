from abc import abstractproperty
from typing import Tuple, Optional, Callable, Iterable, Protocol
from dataclasses import dataclass
from abc import abstractmethod
import numpy as np
from jaxtyping import Float32, Int32
from .distortion import Distortions
from .utils import get_rays


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


@dataclass
class MethodInfo:
    loaded_step: Optional[int] = None
    num_iterations: Optional[int] = None
    batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None


class Method(Protocol):
    @abstractproperty
    def info(self) -> MethodInfo:
        """
        Get method defaults for the trainer.

        Returns:
            Method info.
        """
        return MethodInfo()

    @abstractmethod
    def render(self,
               poses: Float32[np.ndarray, "batch 3 4"],
               intrinsics: Float32[np.ndarray, "batch 3 4"],
               sizes: Tuple[int, int],
               nears_fars: Float32[np.ndarray, "batch 2"],
               distortions: Optional[Distortions] = None,
               progress_callback: Optional[ProgressCallback] = None) -> Iterable[Float32[np.ndarray, "h w c"]]:
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
                    poses: Float32[np.ndarray, "images 3 4"],
                    intrinsics: Float32[np.ndarray, "images 3 4"],
                    sizes: Int32[np.ndarray, "images 2"],
                    nears_fars: Float32[np.ndarray, "batch 2"],
                    images: Float32[np.ndarray, "images h w c"],
                    num_iterations: int,
                    sampling_masks: Optional[Float32[np.ndarray, "images h w"]] = None,
                    distortions: Optional[Distortions] = None):
        """
        Setup training data, model, optimizer, etc.

        Args:
            poses: Camera poses.
            intrinsics: Camera intrinsics.
            sizes: Image sizes.
            images: Images. If the number of channels is 4, the last channel is used as the alpha channel.
            num_iterations: Number of iterations to train.
            sampling_masks: Masks for sampling rays.
            distortions: Distortions.
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
        self.train_poses = None
        self.train_intrinsics = None
        self.train_sizes = None
        self.train_nears_fars = None
        self.train_images = None
        self.train_sampling_masks = None
        self.train_distortions = None
        self.num_iterations = None
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def render_rays(self,
                    origins: Float32[np.ndarray, "batch 3"],
                    directions: Float32[np.ndarray, "batch 3"],
                    nears_fars: Float32[np.ndarray, "batch 3"]):
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
                             origins: Float32[np.ndarray, "batch 3"],
                             directions: Float32[np.ndarray, "batch 3"],
                             nears_fars: Float32[np.ndarray, "batch 3"],
                             colors: Float32[np.ndarray, "batch c"]):
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

    def setup_train(self,
                    poses: Float32[np.ndarray, "images 3 4"],
                    intrinsics: Float32[np.ndarray, "images 3 4"],
                    sizes: Int32[np.ndarray, "images 2"],
                    nears_fars: Float32[np.ndarray, "batch 2"],
                    images: Float32[np.ndarray, "images h w c"],
                    num_iterations: int,
                    sampling_masks: Optional[Float32[np.ndarray, "images h w"]] = None,
                    distortions: Optional[Distortions] = None):
        self.train_poses = poses
        self.train_intrinsics = intrinsics
        self.train_sizes = sizes
        self.train_nears_fars = nears_fars
        self.train_images = images
        self.train_sampling_masks = sampling_masks
        self.train_distortions = distortions
        self.num_iterations = num_iterations

    def train_iteration(self, step: int):
        camera_indices = self._rng.randint(0, len(self.train_images), (self.batch_size,), dtype=np.int32)
        wh = self.train_sizes[camera_indices]
        areas = wh.prod(-1)
        local_indices = np.clip(np.floor(self._rng.rand((self.batch_size,)) * areas), None, areas).astype(np.int32)
        xy = np.stack([local_indices % wh[..., 0], local_indices // wh[..., 0]], -1)
        nears_fars = self.train_nears_fars[camera_indices]
        origins, directions = get_rays(self.train_poses[camera_indices],
                                       self.train_intrinsics[camera_indices],
                                       xy,
                                       self.train_distortions[camera_indices] if self.train_distortions is not None else None)
        colors = self.train_images[camera_indices, xy[..., 1], xy[..., 0]]
        return self.train_iteration_rays(step=step, origins=origins, directions=directions, nears_fars=nears_fars, colors=colors)

    def render(self,
               poses: Float32[np.ndarray, "batch 3 4"],
               intrinsics: Float32[np.ndarray, "batch 3 4"],
               sizes: Int32[np.ndarray, "batch 2"],
               nears_fars: Float32[np.ndarray, "batch 2"],
               distortions: Optional[Distortions] = None,
               progress_callback: Optional[ProgressCallback] = None) -> Iterable[Float32[np.ndarray, "h w c"]]:
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
