from ._types import (
    Method as Method,
    ModelInfo as ModelInfo,
    MethodInfo as MethodInfo,
    RenderOutput as RenderOutput,
    RenderOutputType as RenderOutputType,
    RenderOptions as RenderOptions,
    OptimizeEmbeddingOutput as OptimizeEmbeddingOutput,
    Cameras as Cameras,
    ColorSpace as ColorSpace,
    CameraModel as CameraModel,
    DatasetFeature as DatasetFeature,
    camera_model_to_int as camera_model_to_int,
    camera_model_from_int as camera_model_from_int,
    GenericCameras as GenericCameras,
    Cameras as Cameras,
    new_cameras as new_cameras,
    UnloadedDataset as UnloadedDataset,
    Dataset as Dataset,
    EvaluationProtocol as EvaluationProtocol,
    LicenseSpec as LicenseSpec,
    DatasetLoaderSpec as DatasetLoaderSpec,
    DatasetSpecMetadata as DatasetSpecMetadata,
    LoadDatasetFunction as LoadDatasetFunction,
    DownloadDatasetFunction as DownloadDatasetFunction,
    TrajectoryFrameAppearance as TrajectoryFrameAppearance,
    TrajectoryFrame as TrajectoryFrame,
    Trajectory as Trajectory,
    LoggerEvent as LoggerEvent,
    Logger as Logger,
    OutputArtifact as OutputArtifact,
    ImplementationStatus as ImplementationStatus,
    MethodSpec as MethodSpec,
    DatasetSpec as DatasetSpec,
    LoggerSpec as LoggerSpec,
    EvaluationProtocolSpec as EvaluationProtocolSpec,
    BackendName as BackendName,
    DatasetNotFoundError as DatasetNotFoundError,
    new_dataset as new_dataset,
)
from ._constants import (
    NB_PREFIX as NB_PREFIX,
)
from ._registry import (
    register as register,
    get_method_spec as get_method_spec,
    get_dataset_spec as get_dataset_spec,
    get_logger_spec as get_logger_spec,
    get_evaluation_protocol_spec as get_evaluation_protocol_spec,
    get_dataset_loader_spec as get_dataset_loader_spec,
    get_supported_methods as get_supported_methods,
    get_supported_datasets as get_supported_datasets,
    get_supported_loggers as get_supported_loggers,
    get_supported_evaluation_protocols as get_supported_evaluation_protocols,
    get_supported_dataset_loaders as get_supported_dataset_loaders,
)
from ._method_utils import (
    build_method_class as build_method_class,
    load_checkpoint as load_checkpoint,
)

# We require the __version__ import - the package needs to be installed
try:
    from ._version import __version__  # noqa
except ImportError:
    import sys
    print("Failed to import version from _version.py. Make sure the package was installed correctly by following the official instructions.", flush=True, file=sys.stderr)
    del sys
    __version__ = "develop"
