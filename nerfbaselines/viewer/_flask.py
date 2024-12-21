import queue
import contextlib
import logging
import tempfile
import os
import time
import struct
import json
import io
import threading
import numpy as np
from PIL import Image
import nerfbaselines
from nerfbaselines.results import get_dataset_info
from nerfbaselines.datasets import dataset_index_select, dataset_load_features
from nerfbaselines.utils import image_to_srgb


@contextlib.contextmanager
def _make_render_fn(request_queue, output_queue):
    output_queues = {}
    message_counter = threading.local()
    message_counter.counter = 0

    def render_fn(feedid, **kwargs):
        mid = message_counter.counter
        message_counter.counter += 1
        tid = threading.get_ident()
        feedid = f"{tid}-{mid}"

        if feedid not in output_queues:
            output_queues[feedid] = queue.Queue()
        try:
            request_queue.put({
                "feedid": feedid,
                **kwargs,
            })
            out = output_queues[feedid].get()
            out_type = out.pop("type")
            if out_type == "error":
                raise RuntimeError(out["error"])
            elif out_type == "end":
                raise SystemExit(0)
            return out["result"]
        finally:
            output_queues.pop(feedid)

    # Start queue multiplexer
    def _multiplex_queues(output_queue, output_queues):
        try:
            while True:
                out = output_queue.get()
                out_type = out.get("type")
                if out_type == "end":
                    for q in output_queues.values(): q.put(out)
                else:
                    feedid = out.pop("feedid")
                    assert feedid is not None
                    output_queues[feedid].put(out)
        except Exception as e:
            logging.exception(e)
            logging.error(e)

    multiplex_thread = threading.Thread(target=_multiplex_queues, args=(output_queue, output_queues), daemon=True)
    multiplex_thread.start()
    try:
        yield render_fn
    finally:
        multiplex_thread.join()




def create_ply_bytes(points3D_xyz, points3D_rgb=None):
    from plyfile import PlyData, PlyElement

    # Check if points3D_xyz is a valid array
    if points3D_xyz is None or points3D_xyz.ndim != 2 or points3D_xyz.shape[1] != 3:
        raise ValueError("points3D_xyz must be a 2D array with shape [N, 3].")
    
    # Optional: Check if points3D_rgb is valid if provided
    if points3D_rgb is not None:
        if points3D_rgb.ndim != 2 or points3D_rgb.shape[1] != 3 or points3D_rgb.shape[0] != points3D_xyz.shape[0]:
            raise ValueError("points3D_rgb must be a 2D array with shape [N, 3] matching points3D_xyz.")

    # Create a structured array
    N = points3D_xyz.shape[0]
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if points3D_rgb is not None:
        dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    vertex_data = np.zeros(N, dtype=dtype)
    vertex_data['x'] = points3D_xyz[:, 0]
    vertex_data['y'] = points3D_xyz[:, 1]
    vertex_data['z'] = points3D_xyz[:, 2]

    if points3D_rgb is not None:
        vertex_data['red'] = points3D_rgb[:, 0]
        vertex_data['green'] = points3D_rgb[:, 1]
        vertex_data['blue'] = points3D_rgb[:, 2]

    # Create the PlyElement
    vertex_element = PlyElement.describe(vertex_data, "vertex")

    # Write to a PLY file in memory
    plydata = PlyData([vertex_element])
    output = io.BytesIO()
    plydata.write(output)
    output.seek(0)
    return output



def run_flask_server(request_queue,
                     output_queue, 
                     *,
                     datasets,
                     port,
                     model_info=None,
                     nb_info=None,
                     dataset_metadata=None):
    from flask import Flask, request, jsonify, render_template, Response, send_file
    from flask import request, Response, send_from_directory
    from ._websocket import flask_websocket_route

    # Create app
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    # Reduce logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    _feed_states = {}
    _render_video_tasks = {}

    # Full exception details
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error(f"Server Error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later."
        }), 500
    del handle_exception

    @app.errorhandler(500)
    def handle_500_error(e):
        app.logger.error(f"Server Error: {e}", exc_info=True)  # Log the error with traceback
        return jsonify({
            "status": "error",
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later."
        }), 500
    del handle_500_error

    @app.route("/")
    def index():
        return render_template("index.html")
    del index

    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(os.path.join(app.root_path, 'static'),
                                   'favicon.ico', mimetype='image/vnd.microsoft.icon')
    del favicon

    def render(req):
        if "image_size" not in req:
            raise ValueError("Invalid request, missing image_size")
        if "pose" not in req:
            raise ValueError("Invalid request, missing pose")
        if "intrinsics" not in req:
            raise ValueError("Invalid request, missing intrinsics")
        if "output_type" not in req:
            raise ValueError("Invalid request, missing output_type")
        image_size = tuple(map(int, req.get("image_size").split(",")))
        pose = np.array(list(map(float, req.get("pose").split(","))), dtype=np.float32).reshape(3, 4)
        intrinsics = np.array(list(map(float, req.get("intrinsics").split(","))), dtype=np.float32)
        frame = render_fn(threading.get_ident(), 
                                pose=pose, 
                                image_size=image_size, 
                                intrinsics=intrinsics,
                                output_type=req.get("output_type"),
                                appearance_weights=req.get("appearance_weights"),
                                appearance_train_indices=req.get("appearance_train_indices"),
                                split_percentage=float(req.get("split_percentage", 0.5)),
                                split_tilt=float(req.get("split_tilt", 0)),
                                split_output_type=req.get("split_output_type"))

        with io.BytesIO() as output, Image.fromarray(frame) as img:
            img.save(output, format="JPEG")
            output.seek(0)
            frame_bytes = output.getvalue()
        return frame_bytes

    images_cache = {}

    def get_dataset_image(req):
        if "split" not in req:
            raise ValueError("Invalid request, missing split")
        if "idx" not in req:
            raise ValueError("Invalid request, missing idx")
        idx = int(req["idx"])
        split = req["split"]
        max_img_size = req.get("max_img_size")
        if (split, idx, max_img_size) in images_cache:
            return images_cache[(split, idx, max_img_size)]

        if datasets.get(split) is None:
            return jsonify({"status": "error", "message": "Dataset not found"}), 404
        dataset = datasets[split]
        if not (0 <= idx < len(dataset["cameras"])): 
            return jsonify({"status": "error", "message": "Image not found in the dataset"}), 404

        dataset_slice = dataset_load_features(dataset_index_select(dataset, [idx]), show_progress=False)
        image = dataset_slice["images"][0]

        will_cache = False
        if max_img_size is not None:
            W, H = image.shape[:2]
            downsample_factor = max(1, min(W//int(max_img_size), H//int(max_img_size)))
            image = image[::downsample_factor, ::downsample_factor]
            if max_img_size < 200:
                will_cache = True

        image = image_to_srgb(image, 
                              dtype=np.uint8, 
                              color_space="srgb", 
                              background_color=(dataset.get("metadata") or {}).get("background_color"))
        with io.BytesIO() as output:
            Image.fromarray(image).save(output, format="JPEG")
            output.seek(0)
            out = output.getvalue()
        if will_cache:
            images_cache[(split, idx, max_img_size)] = out
        return out


    def _handle_websocket_message(req):
        thread = req.pop("thread")
        try:
            type = req.pop("type")
            if type == "render":
                frame_bytes = render(req)
                return {"status": "ok", "payload": frame_bytes, "thread": thread}
            elif type == "get_dataset_image":
                image_bytes = get_dataset_image(req)
                return {"status": "ok", "payload": image_bytes, "thread": thread}
            else:
                raise ValueError(f"Invalid message type: {type}")
        except Exception as e:
            return {"status": "error", "message": str(e), "thread": thread}

    @flask_websocket_route(app, "/websocket")
    def websocket(ws):
        while True:
            reqdata = ws.receive()
            req = json.loads(reqdata)
            msg = _handle_websocket_message(req)
            payload = msg.pop("payload", None)
            if payload is None:
                ws.send(json.dumps(msg))
            else:
                msg_bytes = json.dumps(msg).encode("utf-8")
                message_length = len(msg_bytes)
                ws.send(struct.pack(f"!I", message_length) + msg_bytes + payload)
    del websocket

    @app.route("/info", methods=["GET"])
    def info():
        info = {
            "status": "running",
            "method_info": None,
            "dataset_info": None,
            "dataset_parts": [],
        }
        if model_info is not None:
            info["renderer_websocket_url"] = f"./websocket"
        if datasets.get("train") is not None:
            info["dataset_url"] = f"{request.url_root}dataset"

        dataset_metadata_ = dataset_metadata or {}
        if dataset_metadata_.get("viewer_transform") is not None:
            info["viewer_transform"] = dataset_metadata_["viewer_transform"][:3, :4].flatten().tolist()
        if dataset_metadata_.get("viewer_initial_pose") is not None:
            info["viewer_initial_pose"] = dataset_metadata_["viewer_initial_pose"][:3, :4].flatten().tolist()
        if model_info is not None:
            info["output_types"] = model_info.get("supported_outputs", ("color",))

        if datasets.get("train") is not None:
            info["dataset_parts"].append("train")
            if datasets["train"].get("points3D_xyz") is not None:
                info["dataset_parts"].append("pointcloud")

        if dataset_metadata_:
            info["dataset_info"] = info["dataset_info"] or {}
            info["dataset_info"].update({
                k: v.tolist() if hasattr(v, "tolist") else v for k, v in dataset_metadata_.items()
                if not k.startswith("viewer_")
            })

        if dataset_metadata_.get("id") is not None:
            # Add dataset info
            dataset_info = None
            try:
                dataset_info = get_dataset_info(dataset_metadata_["id"])
            except Exception:
                # Perhaps different NB version or custom dataset
                pass
            if dataset_info is not None:
                info["dataset_info"].update(dataset_info)

        if datasets.get("test") is not None:
            info["dataset_parts"].append("test")
        if model_info is not None:
            info["method_info"] = info["method_info"] or {}
            info["method_info"]["method_id"] = model_info["method_id"]
            info["method_info"]["hparams"] = model_info.get("hparams", {})
            if model_info.get("num_iterations") is not None:
                info["method_info"]["num_iterations"] = model_info["num_iterations"]
            if model_info.get("loaded_step") is not None:
                info["method_info"]["loaded_step"] = model_info["loaded_step"]
            if model_info.get("loaded_checkpoint") is not None:
                info["method_info"]["loaded_checkpoint"] = model_info["loaded_checkpoint"]
            if model_info.get("supported_camera_models") is not None:
                info["method_info"]["supported_camera_models"] = list(sorted(model_info["supported_camera_models"]))
            if model_info.get("supported_outputs") is not None:
                info["method_info"]["supported_outputs"] = list(sorted(model_info["supported_outputs"]))

            # Pull more details from the registry
            spec = None
            try:
                spec = nerfbaselines.get_method_spec(model_info["method_id"])
            except Exception:
                pass
            info["method_info"].update(spec.get("metadata") or {})
        if nb_info is not None:
            # Fill in config_overrides, presets, nb_version, and others
            pass
        return jsonify(info)
    del info

    @app.route("/get-state", methods=["POST", "GET"])
    def get_state():
        poll = request.args.get("poll")
        feedid = request.args.get("feedid")
        assert poll is not None and feedid is not None
        if feedid not in _feed_states:
            return jsonify({"status": "error", "message": "feedid not found"}), 404

        # Delay here
        start = time.time()
        while feedid in _feed_states and _feed_states[feedid].version <= int(poll):
            # TODO: Wait condition
            time.sleep(0.01)
            if time.time() - start > 5:
                break

        return jsonify({
            "feedid": feedid,
            "version": _feed_states.get(feedid).version,
            **_feed_states.get(feedid).state,
        })
    del get_state

    @app.route("/render", methods=["POST"])
    def _render_route():
        req = request.json
        try:
            frame_bytes = render(req)
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        return Response(frame_bytes, mimetype="image/jpeg")
    del _render_route

    @app.route("/dataset/pointcloud.ply")
    def dataset_pointcloud():
        if datasets.get("train") is None:
            return jsonify({"status": "error", "message": "Dataset not found"}), 404
        dataset = datasets["train"]
        points3D_xyz = dataset.get("points3D_xyz")
        points3D_rgb = dataset.get("points3D_rgb")
        if points3D_xyz is None:
            return jsonify({"status": "error", "message": "No pointcloud in dataset"}), 404
        output = create_ply_bytes(points3D_xyz, points3D_rgb)
        return send_file(output, download_name="pointcloud.ply", as_attachment=True)
        # return create_and_send_ply(points3D_xyz, points3D_rgb)
    del dataset_pointcloud

    @app.route("/dataset/<string:split>.json")
    def dataset_cameras(split):
        if datasets.get(split) is None:
            return jsonify({"status": "error", "message": "Dataset not found"}), 404
        nb_cameras = datasets[split].get("cameras")
        root_path = datasets[split].get("image_paths_root")
        if root_path is None:
            root_path = os.path.commonpath(datasets[split]["image_paths"])
        cameras = [{
            "pose": pose[:3, :4].flatten().tolist(),
            "intrinsics": intrinsics.tolist(),
            "image_size": image_size.tolist(),
            "image_name": os.path.relpath(image_path, root_path)
        } for _, (pose, intrinsics, image_size, image_path) in enumerate(zip(
            nb_cameras.poses, 
            nb_cameras.intrinsics, 
            nb_cameras.image_sizes,
            datasets[split]["image_paths"]))]
        return jsonify({
            "cameras": cameras
        })
    del dataset_cameras

    @app.route("/dataset/images/<string:split>/<int:idx>.jpg")
    def _get_dataset_image_route(split, idx):
        try:
            max_img_size = request.args.get("size")
            if max_img_size is not None and max_img_size != "":
                max_img_size = int(max_img_size)
            else:
                max_img_size = None
            image_bytes = get_dataset_image({"split": split, "idx": idx, "max_img_size": max_img_size})
        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        return Response(image_bytes, mimetype="image/jpeg")
    del _get_dataset_image_route

    @app.route("/video/<string:videoid>", methods=["PUT"])
    def setup_video(videoid):
        if videoid not in _render_video_tasks:
            _render_video_tasks[videoid] = {
                "status": "pending",
                "progress": 0,
            }
        task = _render_video_tasks[videoid]
        task["trajectory"] = request.json
        return jsonify({ "status": "ok" })
    del setup_video

    @app.route("/video/<string:videoid>.mp4", methods=["GET"])
    def render_video(videoid):
        if videoid not in _render_video_tasks:
            _render_video_tasks[videoid] = {
                "status": "pending",
                "progress": 0,
            }
        task = _render_video_tasks[videoid]
        trajectory = task["trajectory"]
        output_type = "color"
        # First, we start the rendering if not started yet.
        def generate():
            try:
                import mediapy
                yield b""
                startTime = time.time()
                while task.get("trajectory") is None:
                    if time.time() - startTime > 8:
                        raise RuntimeError("Timeout expired waiting for the video to be created")
                    # Waiting for the video stream to be created
                    time.sleep(0.1)
                    yield b""

                with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                    width, height = trajectory["image_size"]
                    with mediapy.VideoWriter(f.name, (height, width), fps=trajectory["fps"], codec="h264") as video:
                        task["status"] = "running"
                        task["progress"] = 0
                        for i, frame in enumerate(trajectory["frames"]):
                            pose = np.array(frame["pose"], dtype=np.float32).reshape(3, 4)
                            intrinsics = np.array(frame["intrinsics"], dtype=np.float32)
                            task["progress"] = i / len(trajectory["frames"])
                            output = render_fn(
                                threading.get_ident(),
                                pose=pose, 
                                image_size=(width, height), 
                                intrinsics=intrinsics,
                                output_type=output_type,
                                split_percentage=0,
                                split_tilt=0,
                                split_output_type=None)
                            video.add_image(output)
                        task["progress"] = 1
                        task["status"] = "done"

                    # Stream generated file
                    f.seek(0)
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                task["status"] = "error"
                task["progress"] = 1
                task["progress_message"] = str(e)
                raise
        return Response(
            generate(), 
            mimetype="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=video.mp4"})
    del render_video

    @app.route("/video-progress/<string:videoid>", methods=["POST", "GET"])
    def get_progress(videoid):
        if videoid not in _render_video_tasks:
            _render_video_tasks[videoid] = { 
                "status": "pending",
                "progress": 0,
            }

        task = _render_video_tasks[videoid]

        def generate():
            last_msg = None
            while task["status"] == "running" or task["status"] == "pending":
                msg = json.dumps({
                    "status": task["status"],
                    "progress": task["progress"],
                })
                if msg != last_msg:
                    yield f'data: {msg}\n\n'
                    last_msg = msg
                time.sleep(0.01)
            msg = json.dumps({
                "status": task["status"],
                "progress": task["progress"],
                "message": task.get("progress_message"),
            })
            yield f'data: {msg}\n\n'
        return Response(generate(), mimetype="text/event-stream")
    del get_progress

    @app.route("/video-feed")
    def video_feed():
        feedid = request.args.get("feedid")
        assert feedid is not None and feedid in _feed_states
        feed = _feed_states[feedid]
        def generate():
            global outputFrame, lock
            # loop over frames from the output stream
            while True:
                frame_bytes = feed.render()
                time.sleep(1/30)
                # yield the output frame in the byte format
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame_bytes) + b'\r\n')
        return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")
    del video_feed

    with _make_render_fn(request_queue, output_queue) as render_fn:
        try:
            app.run(host="0.0.0.0", port=port)
        except Exception as e:
            logging.exception(e)
