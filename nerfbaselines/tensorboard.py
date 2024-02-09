# Disable import not in top level
# pylint: disable=wrong-import-position
import io
from pathlib import Path
from typing import Union

from PIL import Image
import numpy as np
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata  # noqa: F401
from tensorboard.compat.proto.event_pb2 import Event  # noqa: F401
from tensorboard.compat.proto.tensor_pb2 import TensorProto  # noqa: F401
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto  # noqa: F401
from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData  # noqa: F401
from tensorboard.plugins.image.metadata import create_summary_metadata  # noqa: F401
from tensorboard.summary.writer.event_file_writer import EventFileWriter


from .types import Optional
from .utils import convert_image_dtype


class TensorboardWriterEvent:
    def __init__(self, writer, step):
        self._writer = writer
        self.step = step
        self._summaries = []

    def add_image(
        self,
        tag: str,
        image: np.ndarray,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
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
        assert isinstance(value, (float, int))
        self._summaries.append(Summary.Value(tag=tag, simple_value=value))

    def commit(self):
        summary = Summary(value=self._summaries)
        self._writer.add_event(Event(summary=summary, step=self.step))
        self._summaries = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.commit()


class TensorboardWriter:
    def __init__(self, output: Union[str, Path]):
        self._writer = EventFileWriter(str(output))

    def add_event(self, step: int):
        return TensorboardWriterEvent(self._writer, step)

    def add_scalar(self, tag: str, value: Union[float, int], step: int):
        with self.add_event(step) as event:
            event.add_scalar(tag, value)

    def add_text(self, tag: str, text: str, step: int):
        with self.add_event(step) as event:
            event.add_text(tag, text)

    def add_image(
        self,
        tag: str,
        image: np.ndarray,
        step: int,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> None:
        with self.add_event(step) as event:
            event.add_image(tag, image, display_name, description, **kwargs)
