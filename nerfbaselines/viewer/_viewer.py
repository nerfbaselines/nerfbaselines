import io
import time
import queue
import math
import multiprocessing
import logging
import numpy as np
import nerfbaselines
from PIL import Image
from nerfbaselines import __version__
from nerfbaselines.utils import image_to_srgb, apply_colormap, convert_image_dtype
from nerfbaselines.results import get_dataset_info, get_method_info_from_spec
from nerfbaselines import backends
try:
    from typing import Optional
except ImportError:
    from typing_extensions import Optional
from ._httpserver import run_simple_http_server


def get_viewer_params_from_nb_info(nb_info, include_registry_data: bool = True):
    if not nb_info: return {}
    model_info = nb_info.copy()
    dataset_metadata = nb_info.pop("dataset_metadata", {})
    info = get_viewer_params_from_dataset_metadata(dataset_metadata, include_registry_data=include_registry_data)
    method_id = model_info.pop("method", None)

    if method_id is not None and include_registry_data:
        # Pull more details from the registry
        method_info = get_method_info_from_spec(
            nerfbaselines.get_method_spec(method_id))
        # Update with default model info
        info = merge_viewer_params(
            info,
            get_viewer_params_from_model_info(method_info, include_registry_data=include_registry_data))

    # Finally, update with nb_info itself
    info["state"].setdefault("method_info", {}).update(_fix_types(model_info))
    info["method_id"] = method_id
    info.pop("renderer", None)
    return info
    

def _fix_types(x):
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_fix_types(y) for y in x]
    elif isinstance(x, dict):
        return {k: _fix_types(v) for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        return x.tolist()
    return x


def get_viewer_params_from_dataset_metadata(dataset_metadata, include_registry_data: bool = True):
    if not dataset_metadata: return {}
    info = {"state": {}}
    if dataset_metadata.get("viewer_transform") is not None:
        viewer_transform = dataset_metadata["viewer_transform"]
        if isinstance(viewer_transform, np.ndarray):
            viewer_transform = viewer_transform[:3, :4].flatten().tolist()
        info["viewer_transform"] = viewer_transform
    if dataset_metadata.get("viewer_initial_pose") is not None:
        viewer_initial_pose = dataset_metadata["viewer_initial_pose"]
        if isinstance(viewer_initial_pose, np.ndarray):
            viewer_initial_pose = viewer_initial_pose[:3, :4].flatten().tolist()
        info["viewer_initial_pose"] = viewer_initial_pose
    if dataset_metadata.get("depth_range") is not None:
        # Add output range configuration
        info["state"]["outputs_configuration"] = {
            "depth": {
                "range_min": dataset_metadata["depth_range"][0],
                "range_max": dataset_metadata["depth_range"][1],
            },
        }

    info["state"]["dataset_info"] = _dataset_info = {}
    _dataset_info.update({
        k: v.tolist() if hasattr(v, "tolist") else v for k, v in dataset_metadata.items()
        if not k.startswith("viewer_")
    })

    if dataset_metadata.get("id") is not None and include_registry_data:
        # Add dataset info
        try:
            dataset_info = get_dataset_info(dataset_metadata["id"]).copy()
            _dataset_info.update(_fix_types(dataset_info))
        except Exception:
            # Perhaps different NB version or custom dataset
            pass
    return info


def get_viewer_params_from_dataset(train_dataset, test_dataset, include_registry_data: bool = True):
    info = {}
    if train_dataset is not None or test_dataset is not None:
        info.setdefault("dataset", {})
        info["dataset"]["url"] = "./dataset.json"
        if train_dataset is not None and train_dataset.get("points3D_xyz") is not None:
            info["dataset"]["pointcloud_url"] = "./dataset/pointcloud.ply"

    if train_dataset and train_dataset.get("metadata") is not None:
        dataset_metadata = train_dataset["metadata"]
        info.update(get_viewer_params_from_dataset_metadata(dataset_metadata, include_registry_data=include_registry_data))
    return info


def get_viewer_params_from_model_info(model_info, include_registry_data: bool = True):
    if not model_info: return {}
    info = {"state":{}}
    output_types = model_info.get("supported_outputs", ("color",))
    output_types = [x if isinstance(x, str) else x["name"] for x in output_types]
    info["renderer"] = {
        "type": "remote",
        "websocket_url": "./render-websocket",
        "http_url": "./render",
        "output_types": output_types,
    }

    if "viewer_default_resolution" in model_info:
        default_resolution = model_info["viewer_default_resolution"]
        if isinstance(default_resolution, int):
            info["state"]["render_resolution"] = default_resolution
            info["state"]["prerender_enabled"] = False
        elif isinstance(default_resolution, (list, tuple)) and len(default_resolution) == 2:
            prerender_resolution, default_resolution = default_resolution
            info["state"]["render_resolution"] = default_resolution
            info["state"]["prerender_resolution"] = prerender_resolution
            info["state"]["prerender_enabled"] = True
    info["state"]["method_info"] = _method_info = {}
    _method_info["method_id"] = model_info["method_id"]
    _method_info["hparams"] = model_info.get("hparams", {})
    if model_info.get("num_iterations") is not None:
        _method_info["num_iterations"] = model_info["num_iterations"]
    if model_info.get("loaded_step") is not None:
        _method_info["loaded_step"] = model_info["loaded_step"]
    if model_info.get("loaded_checkpoint") is not None:
        _method_info["loaded_checkpoint"] = model_info["loaded_checkpoint"]
    if model_info.get("supported_camera_models") is not None:
        _method_info["supported_camera_models"] = list(sorted(model_info["supported_camera_models"]))
    if model_info.get("supported_outputs") is not None:
        supported_outputs = [x if isinstance(x, str) else x["name"] for x in model_info["supported_outputs"]]
        _method_info["supported_outputs"] = list(sorted(supported_outputs))
        info["state"]["outputs_configuration"] = {}
        for output in model_info["supported_outputs"]:
            name = output if isinstance(output, str) else output["name"]
            tp = output if isinstance(output, str) else output.get("type", name)
            if tp == "depth":
                config = { "palette_enabled": True }
            elif tp == "color" or tp == "normal":
                config = {}
            elif tp == "accumulation":
                config = { "palette_enabled": True, "range_min": 0, "range_max": 1 }
            else:
                config = { "palette_enabled": True }
            info["state"]["outputs_configuration"][name] = config

    # Pull more details from the registry
    if include_registry_data:
        spec = None
        try:
            spec = nerfbaselines.get_method_spec(model_info["method_id"])
        except Exception:
            pass
        if spec is not None:
            metadata = (spec.get("metadata") or {}).copy()
            metadata.pop("paper_results", None)
            info["state"]["method_info"].update(_fix_types(metadata))
    return info


def merge_viewer_params(*args):
    def deepmerge(a, b):
        out = {**a, **b}
        for k in out.keys():
            if k in a and k in b and isinstance(a[k], dict) and isinstance(b[k], dict):
                out[k] = deepmerge(a[k], b[k])
        return out
    if not args: return {}
    if len(args) == 1: return args[0]
    if len(args) > 2:
        return merge_viewer_params(args[0], merge_viewer_params(*args[1:]))
    assert len(args) == 2
    a, b = args
    out = {**a, **b}
    if "renderer" in a or "renderer" in b:
        a_renderer = a.get("renderer", {})
        b_renderer = b.get("renderer", {})
        out["renderer"] = {**a_renderer, **b_renderer}
    if "dataset" in a or "dataset" in b:
        a_dataset = a.get("dataset", {})
        b_dataset = b.get("dataset", {})
        out["dataset"] = {**a_dataset, **b_dataset}
    if "state" in a or "state" in b:
        astate = a.get("state", {})
        bstate = b.get("state", {})
        out["state"] = state = {**astate, **bstate}
        if "method_info" in astate or "method_info" in bstate:
            state["method_info"] = {
                **astate.get("method_info", {}),
                **bstate.get("method_info", {}),
            }
        if "dataset_info" in astate or "dataset_info" in bstate:
            state["dataset_info"] = {
                **astate.get("dataset_info", {}),
                **bstate.get("dataset_info", {}),
            }
        if "outputs_configuration" in astate and "outputs_configuration" in bstate:
            state["outputs_configuration"] = deepmerge(
                astate["outputs_configuration"], bstate["outputs_configuration"]
            )
    return out


def get_viewer_params(model_info, datasets, nb_info):
    return merge_viewer_params(
        get_viewer_params_from_nb_info(nb_info),
        get_viewer_params_from_model_info(model_info),
        get_viewer_params_from_dataset(datasets.get("train"), datasets.get("test")),
    )


def combine_outputs(o1, o2, *, split_percentage=0.5, split_tilt=0):
    """
    Combines two outputs o1 and o2 with a tilted splitting line.
    NOTE: It writes to the first output array in-place.
    
    Args:
        o1 (np.ndarray): First output array.
        o2 (np.ndarray): Second output array.
        split_percentage (float): Percentage along the width where the split starts (0 to 1).
        split_tilt (float): Angle in degrees to tilt the split line (-45 to 45).
    
    Returns:
        np.ndarray: Combined output array with the tilt applied.
    """
    assert o1.shape == o2.shape, f"Output shapes do not match: {o1.shape} vs {o2.shape}"
    height, width = o1.shape[:2]
    split_percentage_ = split_percentage if split_percentage is not None else 0.5

    tilt_radians = split_tilt * math.pi / 180
    split_dir = [math.cos(tilt_radians), math.sin(tilt_radians)]
    y, x = np.indices((height, width))
    proj = (x-width/2) * split_dir[0] + (y-height/2) * split_dir[1]
    split_dir_len = width/2*abs(split_dir[0]) + height/2*abs(split_dir[1])
    split_mask = proj > (split_percentage_*2-1) * split_dir_len

    # Apply the split mask
    result = o1
    result[split_mask] = o2[split_mask]
    return result


class Viewer:
    def __init__(self, 
                 model=None, 
                 train_dataset=None, 
                 test_dataset=None, 
                 nb_info=None, 
                 host: str = "localhost",
                 port: Optional[int] = None):
        self._request_queue = multiprocessing.Queue()
        self._output_queue = multiprocessing.Queue()
        self._port = port
        self._nb_info = nb_info

        # Prepare information
        self._output_types_map = None
        self._model_info = None
        if model is not None:
            self._model_info = model.get_info()
            self._output_types_map = {
                x if isinstance(x, str) else x["name"]: {"name": x, "type": x} if isinstance(x, str) else x for x in self._model_info.get("supported_outputs", ("color",))}
        self._default_background_color = (self._nb_info or {}).get("dataset_metadata", {}).get("background_color", (0, 0, 0))
        self._default_background_color = np.array(self._default_background_color, dtype=np.uint8)
        self._default_expected_depth_scale = (self._nb_info or {}).get("dataset_metadata", {}).get("expected_depth_scale", 0.5)
        self._default_depth_range = (self._nb_info or {}).get("dataset_metadata", {}).get("depth_range", None)
        self._process = None
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._server_fn = run_simple_http_server
        self._model = model
        self._running = False
        self._host = host

    @property
    def port(self):
        return self._port

    def run(self):
        """
        Starts the update loop that continuously processes render tasks.
        The function will block until the viewer is closed by the user.
        """
        while True:
            self.update()

    def update(self) -> bool:
        """
        Performs a single render task from the queue.
        This function should be called within the train step function to update the viewer when rendering is available.

        Returns:
            bool: True if there are more tasks in the queue
        """
        if self._request_queue is None or self._output_queue is None:
            raise RuntimeError("Viewer is closed")
        req = self._request_queue.get()
        feedid = req.pop("feedid")
        try:
            frame = self._render(**req)
            self._output_queue.put({
                "type": "result",
                "feedid": feedid,
                "result": frame,
            })
        except Exception as e:
            logging.exception(e)
            self._output_queue.put({
                "type": "error",
                "feedid": feedid,
                "error": str(e),
                "exception": e
            })
        return not self._request_queue.empty()

    def _format_output(self, output, name, *, background_color=None, expected_depth_scale=None, palette=None, output_range=None):
        name = name or "color"
        if self._output_types_map is None:
            raise ValueError("No output types map available")
        rtype_spec = self._output_types_map.get(name)
        if rtype_spec is None:
            raise ValueError(f"Unknown output type: {name}")
        if background_color is None:
            background_color = self._default_background_color
        if expected_depth_scale is None:
            expected_depth_scale = self._default_expected_depth_scale
        rtype = rtype_spec.get("type", name)
        if rtype == "color":
            output = image_to_srgb(output, np.uint8, color_space="srgb", allow_alpha=False, background_color=background_color)
        elif rtype == "normal":
            output = convert_image_dtype(output*0.5+0.5, np.uint8)
        elif rtype == "depth":
            # Blend depth with correct color pallete
            # mmin, mmax = output.min(), output.max()
            def map_output(x): return 1-1/(x+1)
            mmin = mmax = None
            if output_range is not None:
                mmin, mmax = output_range
            mmin = 0 if mmin is None else map_output(mmin)
            mmax = 1 if mmax is None else map_output(mmax)
            output = map_output(output)
            # Map to a color scale
            output = output * 0 if mmax == mmin else (output-mmin)/(mmax-mmin)
            output = apply_colormap(output, pallete=palette or "coolwarm", invert=True)
        elif rtype == "accumulation":
            output = apply_colormap(output, pallete=palette or "coolwarm")
        elif len(output.shape) == 2:
            mmin, mmax = None, None
            if output_range is not None:
                mmin, mmax = output_range
            if mmin is None: mmin = output.min()
            if mmax is None: mmax = output.max()
            output = apply_colormap((output-mmin)/(mmax-mmin), pallete=palette or "coolwarm")
        return output

    def _render(self,
                pose,
                image_size,
                intrinsics,
                output_type,
                split_output_type,
                split_percentage,
                split_tilt,
                palette=None,
                appearance_weights=None,
                appearance_train_indices=None,
                output_range=None,
                split_range=None,
                split_palette=None,
                format=None,
                **kwargs):
        del kwargs
        camera = nerfbaselines.new_cameras(
            poses=pose,
            intrinsics=np.array(intrinsics, dtype=np.float32),
            camera_models=np.array(0, dtype=np.int32),
            image_sizes=np.array(image_size, dtype=np.int32),
        )
        output_types = (output_type,)
        if split_output_type is not None:
            output_types = output_types + (split_output_type,)
        embedding = None
        embeddings = None
        if self._model is None:
            raise RuntimeError("No model was loaded for rendering")
        if appearance_weights is not None and appearance_train_indices is not None:
            if sum(appearance_weights, start=0) < 1e-6:
                app_tuples = [(0, 0)]
            else:
                app_tuples = [(idx, weight) for idx, weight in zip(appearance_train_indices, appearance_weights) if weight > 0]
            try:
                embeddings = [self._model.get_train_embedding(idx) for idx, _ in app_tuples]
            except AttributeError:
                pass
            except NotImplementedError:
                pass
            if embeddings is not None and all(x is not None for x in embeddings):
                embedding = sum(x * alpha for (_, alpha), x in zip(app_tuples, embeddings))
        options = { "output_type_dtypes": { "color": "uint8" },
                    "outputs": output_types,
                    "embedding": embedding }
        with backends.zero_copy():
            outputs = self._model.render(camera, options=options)
            # Format first output
            frame = self._format_output(outputs[output_type], output_type, 
                                        palette=palette,
                                        output_range=output_range)
            if split_output_type is not None:
                # Format second output
                split_frame = self._format_output(outputs[split_output_type], split_output_type,
                                                  palette=split_palette,
                                                  output_range=split_range)

                # Combine the two outputs
                frame = combine_outputs(frame, split_frame, 
                                        split_percentage=split_percentage,
                                        split_tilt=split_tilt)
            # Copy if still in shared memory before moving it to the output queue
            with io.BytesIO() as output, Image.fromarray(frame) as img:
                img.save(output, format=format)
                output.seek(0)
                return output.getvalue()


    def _run_backend(self):
        orig_port = self._port
        datasets = {"train": self._train_dataset, "test": self._test_dataset}
        info = get_viewer_params(self._model_info, datasets, self._nb_info)
        assert self._request_queue is not None, "Request queue is not initialized"

        self._process = multiprocessing.Process(target=self._server_fn, args=(
            self._request_queue, 
            self._output_queue, 
        ), kwargs=dict(
            datasets={"train": self._train_dataset, "test": self._test_dataset},
            info=info,
            host=self._host,
            port=self._port,
        ), daemon=True)
        self._process.start()

        # Wait for the viewer to start
        while self._process.is_alive():
            try:
                message = self._request_queue.get(timeout=1)
                msg_type = message.get("type")
                if msg_type != "start":
                    logging.error(f"Unexpected message: {message}")
                    raise RuntimeError("Viewer backend process failed to start")
                self._port = message.get("port")
                break
            except queue.Empty:
                pass

        if self._process is None or not self._process.is_alive():
            raise RuntimeError("Viewer backend process failed to start")

        if orig_port != self._port and orig_port is not None and orig_port > 0:
            logging.warning(f"Port {orig_port} is already in use, using port {self._port} instead")
        # Log the viewer url
        logging.info(f"Viewer running at http://{self._get_hostname()}/")

    def _get_hostname(self):
        if ":" in self._host:
            return f"[{self._host}]:{self._port}"
        return f"{self._host}:{self._port}"

    def close(self):
        # Empty request queue
        if self._request_queue is not None:
            while not self._request_queue.empty(): self._request_queue.get()
            self._request_queue = None

        # Signal output queues to stop
        if self._output_queue is not None:
            self._output_queue.put({ "type": "end" })
            self._output_queue = None
        if self._process is not None:
            _start = time.time()
            while self._process.is_alive():
                if time.time() - _start > 5:
                    logging.error("Viewer process did not close in time")
                    break
            self._process.kill()
            self._process = None

    def __enter__(self):
        if self._process is None:
            self._run_backend()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.close()

    def show_in_notebook(self, width: str = "100%", height: str = "600"):
        google_colab_output = None
        try:
            from google.colab import output as google_colab_output  # type: ignore
        except Exception:
            pass
        if google_colab_output is not None:
            logging.debug("Running in Google Colab, returning port")
            google_colab_output.serve_kernel_port_as_iframe(self._port, width=width, height=height)
            return

        import IPython.display
        from IPython.display import display, HTML
        setattr(IPython.display, "_iframe_counter", 1+getattr(IPython.display, "_iframe_counter", 0))
        port = self._port
        iframe_id = f"nb-iframe-{IPython.display._iframe_counter}"  # type: ignore
        display(HTML(f"""
<iframe id="{iframe_id}" width="{width}" height="{height}" allowfullscreen src="http://{self._get_hostname()}/"></iframe>
<script>
(function () {{
const iframe = document.getElementById("{iframe_id}");
fetch("http://{self._get_hostname()}/").then(x => {{ 
  if (!x.ok) throw new Error("Cannot reach default address http://{self._get_hostname()}/"); 
}}).catch(() => {{
  fetch("/proxy/{port}/").then(x => {{ 
    if (!x.ok) throw new Error("Cannot reach proxy address /proxy/{port}/");
    iframe.setAttribute("src", "/proxy/{port}/");
  }}).catch(() => {{
    iframe.setAttribute("srcdoc", `<h1>Failed to load</h1><p>Please make sure the viewer is running. If you run Jupyter remotely, please install jupyter-proxy-server extension to enable port forwarding. VSCode jupyter notebooks currently require the port to be forwarded. Please forward port: {port}.</p>`);
 }});
}});
}})();
</script>
        """))
