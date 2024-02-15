import shutil
import math
import warnings
import os
from PIL import Image
import numpy as np
import io
import contextlib

from pathlib import Path
import typing
from typing import Optional, Union, List, Dict, Sequence, Any
from typing import TYPE_CHECKING
from .utils import convert_image_dtype
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

if TYPE_CHECKING:
    import wandb.sdk.wandb_run


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


class BaseLoggerEvent(LoggerEvent):
    def add_scalar(self, tag: str, value: Union[float, int]) -> None:
        raise NotImplementedError()
    
    def add_text(self, tag: str, text: str) -> None:
        raise NotImplementedError()
    
    def add_image(self, tag: str, image: np.ndarray, display_name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> None:
        raise NotImplementedError()
    
    def add_embedding(self, tag: str, embeddings: np.ndarray, *,
                        images: Optional[List[np.ndarray]] = None, 
                        labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        raise NotImplementedError()
    
    def add_plot(self, tag: str, *data: np.ndarray,
                 axes_labels: Optional[Sequence[str]] = None, 
                 title: Optional[str] = None,
                 colors: Optional[Sequence[np.ndarray]] = None,
                 labels: Optional[Sequence[str]] = None,
                 **kwargs) -> None:
        assert len(data) > 0, "At least one data array should be provided"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        assert all(len(d.shape) == 2 for d in data), "All data should have two dimensions"
        assert all(d.shape[1] == data[0].shape[1] for d in data), "All data should have the same number of columns"
        num_dim = data[0].shape[1]
        if axes_labels is None:
            axes_labels = ["x", "y", "z"][:num_dim]
        else:
            assert num_dim == len(axes_labels), "All data should have the same number of columns as axes_labels"
        assert data[0].shape[1] == 2, "Only 2D plots are supported"

        colors_mpl = None
        if colors is not None:
            assert len(colors) == len(data), "Number of colors should match number of data arrays"
            assert all(c.shape == (3,) for c in colors), "All colors should be RGB"
            colors_mpl = [tuple((c / 255).tolist()) for c in colors]
        
        if labels is not None:
            assert len(labels) == len(data), "Number of labels should match number of data arrays"

        # Render the image using matplotlib
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        for i, d in enumerate(data):
            x, y = d.T
            kwargs = {}
            if colors_mpl is not None:
                kwargs["color"] = colors_mpl[i]
            if labels is not None:
                kwargs["label"] = labels[i]
            ax.plot(x, y, **kwargs)
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])

        # Render plot as np array
        fig.canvas.draw()
        with io.BytesIO() as img_buf:
            fig.savefig(img_buf, format='png')
            img_buf.seek(0)
            plot_img = np.array(Image.open(img_buf))
        plt.close(fig)
        self.add_image(tag, plot_img, display_name=title, description=title)


class BaseLogger(Logger):
    def add_event(self, step: int) -> typing.ContextManager[LoggerEvent]:
        raise NotImplementedError()

    def add_scalar(self, tag: str, value: Union[float, int], step: int):
        with self.add_event(step) as event:
            event.add_scalar(tag, value)

    def add_text(self, tag: str, text: str, step: int):
        with self.add_event(step) as event:
            event.add_text(tag, text)

    def add_image(self, tag: str, image: np.ndarray, step: int, *, display_name: Optional[str] = None, description: Optional[str] = None, **kwargs):
        with self.add_event(step) as event:
            event.add_image(tag, image, display_name, description, **kwargs)

    def add_embedding(self, tag: str, embeddings: np.ndarray, step: int, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        with self.add_event(step) as event:
            event.add_embedding(tag, embeddings, images=images, labels=labels)


class WandbLoggerEvent(BaseLoggerEvent):
    def __init__(self, commit):
        self._commit: Dict[str, Any] = commit

    def add_scalar(self, tag: str, value: Union[float, int]) -> None:
        self._commit[tag] = value

    def add_text(self, tag: str, text: str) -> None:
        self._commit[tag] = text

    def add_image(self, tag: str, image: np.ndarray, display_name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> None:
        import wandb
        self._commit[tag] = [wandb.Image(image, caption=description)]

    def add_embedding(self, tag: str, embeddings: np.ndarray, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
                    
        import wandb
        table = wandb.Table()
        table.add_column("embedding", embeddings)
        if labels is not None:
            if isinstance(labels[0], dict):
                for key in labels[0].keys():
                    table.add_column(key, [label[key] for label in labels])
            else:
                table.add_column("label", labels)
        if images is not None:
            for image in images:
                table.add_column("image", wandb.Image(image))
        self._commit[tag] = table
    

class WandbLogger(BaseLogger):
    def __init__(self, output: Union[str, Path]):
        import wandb
        wandb_run: "wandb.sdk.wandb_run.Run" = wandb.init(dir=str(output))
        self._wandb_run = wandb_run
        self._wandb = wandb

    @contextlib.contextmanager
    def add_event(self, step: int):
        commit = {}
        yield WandbLoggerEvent(commit)
        self._wandb_run.log(commit, step=step)

    def __str__(self):
        return "wandb"


class ConcatLoggerEvent:
    def __init__(self, events):
        self.events = events

    def __getattr__(self, name):
        callbacks = []
        for event in self.events:
            callbacks.append(getattr(event, name))
        def call(*args, **kwargs):
            for callback in callbacks:
                callback(*args, **kwargs)
        return call


class ConcatLogger(BaseLogger):
    def __init__(self, loggers):
        self.loggers = loggers

    def __bool__(self):
        return bool(self.loggers)

    @contextlib.contextmanager
    def add_event(self, step: int):
        def enter_event(loggers, events):
            if loggers:
                with loggers[0].add_event(step) as event:
                    return enter_event(loggers[1:], [event] + events)
            else:
                return ConcatLoggerEvent(events)
        yield enter_event(self.loggers, [])

    def __str__(self):
        if not self:
            return "[]"
        return ",".join(map(str, self.loggers))


class TensorboardLoggerEvent(BaseLoggerEvent):
    def __init__(self, logdir, summaries, step):
        self._step = step
        self._logdir = logdir
        self._summaries = summaries

    @staticmethod
    def _encode(rawstr):
        # I'd use urllib but, I'm unsure about the differences from python3 to python2, etc.
        retval = rawstr
        retval = retval.replace("%", f"%{ord('%'):02x}")
        retval = retval.replace("/", f"%{ord('/'):02x}")
        retval = retval.replace("\\", "%%%02x" % (ord("\\")))
        return retval

    def add_image(
        self,
        tag: str,
        image: np.ndarray,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.plugins.image.metadata import create_summary_metadata

        metadata = None
        if display_name is not None or description is not None:
            metadata = create_summary_metadata(
                display_name=display_name or "",
                description=description or "",
            )
        with io.BytesIO() as simg:
            image = convert_image_dtype(image, np.uint8)
            Image.fromarray(image).save(simg, format="png")
            self._summaries.append(
                Summary.Value(
                    tag=tag,
                    metadata=metadata,
                    image=Summary.Image(
                        encoded_image_string=simg.getvalue(),
                        height=image.shape[0],
                        width=image.shape[1],
                    ),
                )
            )

    def add_text(self, tag: str, text: str) -> None:
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
        from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData

        plugin_data = SummaryMetadata.PluginData(
            plugin_name="text", content=TextPluginData(version=0).SerializeToString()
        )
        smd = SummaryMetadata(plugin_data=plugin_data)
        tensor = TensorProto(
            dtype="DT_STRING",
            string_val=[text.encode("utf8")],
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
        )
        self._summaries.append(Summary.Value(tag=tag, metadata=smd, tensor=tensor))

    def add_scalar(self, tag: str, value: Union[float, int]) -> None:
        from tensorboard.compat.proto.summary_pb2 import Summary

        assert isinstance(value, (float, int))
        self._summaries.append(Summary.Value(tag=tag, simple_value=value))

    def add_embedding(self, tag: str, embeddings: np.ndarray, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
        from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo
        from tensorboard.compat import tf
        from google.protobuf import text_format

        def make_sprite(label_img, save_path):
            # Background white for white tensorboard theme and black for dark theme
            background = 255

            # this ensures the sprite image has correct dimension as described in
            # https://www.tensorflow.org/get_started/embedding_viz
            nrow = int(math.ceil(len(label_img) ** 0.5))
            label_img = [convert_image_dtype(img, np.uint8) for img in label_img]
            mh, mw = max(img.shape[0] for img in label_img), max(img.shape[1] for img in label_img)

            arranged_augment_square_HWC = np.full((mh * nrow, mw * nrow, 3), background, dtype=np.uint8)
            for i, image in enumerate(label_img):
                img_width = ow = image.shape[1]
                img_height = oh = image.shape[0]
                aspect = img_width / img_height
                img_width = int(min(mw, aspect * mh))
                img_height = int(img_width / aspect)
                if img_width != ow or img_height != oh:
                    img = Image.fromarray(image)
                    img = img.resize((img_width, img_height))
                    image = np.array(img)
                x = i % nrow
                y = i // nrow
                h, w = image.shape[:2]
                offx = x * mw + (mw - w) // 2
                offy = y * mh + (mh - h) // 2
                arranged_augment_square_HWC[offy : offy + h, offx : offx + w] = image

            im = Image.fromarray(arranged_augment_square_HWC)
            im.save(os.path.join(str(save_path), "sprite.png"))
            return mw, mh

        # Maybe we should encode the tag so slashes don't trip us up?
        # I don't think this will mess us up, but better safe than sorry.
        subdir = Path(f"{str(self._step).zfill(5)}/{self._encode(tag)}")
        save_path = Path(self._logdir) / subdir

        if save_path.exists():
            shutil.rmtree(str(save_path))
            warnings.warn(f"Removing existing log directory: {save_path}")
        save_path.mkdir(parents=True)

        if labels is not None:
            assert len(labels) == len(embeddings), "#labels should equal with #data points"
            tsv = []
            if len(labels) > 0:
                if isinstance(labels[0], dict):
                    metadata_header = list(labels[0].keys())
                    metadata = [metadata_header] + [[str(x.get(k, "")) for k in metadata_header] for x in labels]
                    tsv = ["\t".join(str(e) for e in ln) for ln in metadata]
                else:
                    metadata = labels
                    tsv = [str(x) for x in metadata]
            metadata_bytes = tf.compat.as_bytes("\n".join(tsv) + "\n")
            with (save_path / "metadata.tsv").open("wb") as f:
                f.write(metadata_bytes)

        if images is not None:
            assert (
                len(images) == embeddings.shape[0]
            ), "#images should equal with #data points"
            label_img_size = make_sprite(images, save_path)

        assert (
            embeddings.ndim == 2
        ), "mat should be 2D, where mat.size(0) is the number of data points"
        with (save_path / "tensors.tsv").open("wb") as f:
            for x in embeddings:
                x = [str(i.item()) for i in x]
                f.write(tf.compat.as_bytes("\t".join(x) + "\n"))

        projector_config = ProjectorConfig()
        if (Path(self._logdir) / "projector_config.pbtxt").exists():
            message_bytes = (Path(self._logdir) / "projector_config.pbtxt").read_bytes()
            projector_config = text_format.Parse(message_bytes, projector_config)

        embedding_info = EmbeddingInfo()
        embedding_info.tensor_name = f"{tag}:{str(self._step).zfill(5)}"
        embedding_info.tensor_path = str(subdir / "tensors.tsv")
        if labels is not None:
            embedding_info.metadata_path = str(subdir / "metadata.tsv")
        if images is not None:
            embedding_info.sprite.image_path = str(subdir / "sprite.png")
            embedding_info.sprite.single_image_dim.extend(label_img_size)
        projector_config.embeddings.extend([embedding_info])

        config_pbtxt = text_format.MessageToString(projector_config)
        with (Path(self._logdir) / "projector_config.pbtxt").open("wb") as f:
            f.write(tf.compat.as_bytes(config_pbtxt))


class TensorboardLogger(BaseLogger):
    def __init__(self, output: Union[str, Path]):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self._writer = EventFileWriter(str(output))

    @contextlib.contextmanager
    def add_event(self, step: int):
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.compat.proto.event_pb2 import Event

        summaries = []
        yield TensorboardLoggerEvent(self._writer.get_logdir(), summaries, step=step)
        summary = Summary(value=summaries)
        self._writer.add_event(Event(summary=summary, step=step))
    
    def __str__(self):
        return "tensorboard"