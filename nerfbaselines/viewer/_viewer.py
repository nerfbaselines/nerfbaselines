import os
from contextlib import ExitStack, contextmanager
import multiprocessing
import threading
import copy
import queue
import time
import logging
import uuid
import io
from flask import Flask, request, jsonify, render_template, Response, send_file
import numpy as np
from PIL import Image, ImageDraw
import nerfbaselines

pcs = set()
candidates = []

TEMPLATES_AUTO_RELOAD = True


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


def bouncing_ball_frame(time, width=400, height=400, **kwargs):
    del kwargs
    # Parameters
    ball_radius = 20
    ball_color = (255, 0, 0)  # Red
    background_color = (0, 0, 0)  # Black
    vx, vy = 3, 2  # Velocity of the ball (pixels per time unit)

    # Compute the position of the ball
    x = (vx * time) % (2 * (width - ball_radius))  # Horizontal motion
    y = (vy * time) % (2 * (height - ball_radius))  # Vertical motion

    # Reflect motion at boundaries
    x = 2 * (width - ball_radius) - x if x > (width - ball_radius) else x
    y = 2 * (height - ball_radius) - y if y > (height - ball_radius) else y

    # Create image
    img = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        (x - ball_radius, y - ball_radius, x + ball_radius, y + ball_radius),
        fill=ball_color
    )
    frame = np.array(img)[:, :, ::-1]
    return frame


def _run_viewer_server(request_queue, output_queue, *args, **kwargs):
    output_queues = {}

    def render_fn(feedid, **kwargs):
        if feedid not in output_queues:
            output_queues[feedid] = queue.Queue()
        request_queue.put({
            "type": "render",
            "feedid": feedid,
            **kwargs,
        })
        out = output_queues[feedid].get()
        out_type = out.pop("type")
        if out_type == "error":
            raise RuntimeError(out["error"]) from out["exception"]
        elif out_type == "end":
            raise SystemExit(0)
        return out["result"]

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

    app = _build_flake_app(render_fn, *args, **kwargs)
    multiplex_thread = threading.Thread(target=_multiplex_queues, args=(output_queue, output_queues), daemon=True)
    multiplex_thread.start()
    try:
        app.run(host="0.0.0.0", port=5001)
    finally:
        multiplex_thread.join()


def _build_flake_app(render_fn, 
                     datasets,
                     dataset_metadata=None):
    # Create debug app
    app = Flask(__name__)
    # Set it to debug mode
    app.debug = True
    _feed_states = {}

    @app.errorhandler(500)
    def handle_500_error(e):
        app.logger.error(f"Server Error: {e}", exc_info=True)  # Log the error with traceback
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later."
        }), 500
    del handle_500_error

    @app.route("/")
    def index():
        return render_template("index.html")
    del index

    @app.route("/info", methods=["GET"])
    def info():
        info = {
            "status": "running",
            "method": {},
        }
        dataset_metadata_ = dataset_metadata or {}
        if dataset_metadata_.get("viewer_transform") is not None:
            info["viewer_transform"] = dataset_metadata_["viewer_transform"][:3, :4].flatten().tolist()
        if dataset_metadata_.get("viewer_initial_pose") is not None:
            info["viewer_initial_pose"] = dataset_metadata_["viewer_initial_pose"][:3, :4].flatten().tolist()
        info["output_types"] = ["color", "depth"]
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
    def render():
        req = request.args
        image_size = tuple(map(int, req.get("image_size").split(",")))
        pose = np.array(list(map(float, req.get("pose").split(","))), dtype=np.float32).reshape(3, 4)
        intrinsics = np.array(list(map(float, req.get("intrinsics").split(","))), dtype=np.float32)
        frame_bytes = render_fn(threading.get_ident(), 
                                pose=pose, 
                                image_size=image_size, 
                                intrinsics=intrinsics)
        return Response(frame_bytes, mimetype="image/jpeg")
    del render

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
        # return send_file(output, mimetype="application/octet-stream")
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
    def dataset_images(split, idx):
        from nerfbaselines.datasets import dataset_index_select, dataset_load_features
        from nerfbaselines.utils import image_to_srgb

        if datasets.get(split) is None:
            return jsonify({"status": "error", "message": "Dataset not found"}), 404
        dataset = datasets[split]
        if not (0 <= idx < len(dataset["cameras"])): 
            return jsonify({"status": "error", "message": "Image not found in the dataset"}), 404

        dataset_slice = dataset_load_features(dataset_index_select(dataset, [idx]))
        image = dataset_slice["images"][0]

        max_img_size = request.args.get("size")
        if max_img_size is not None:
            W, H = image.shape[:2]
            downsample_factor = max(1, min(W//int(max_img_size), H//int(max_img_size)))
            image = image[::downsample_factor, ::downsample_factor]

        image = image_to_srgb(image, 
                              dtype=np.uint8, 
                              color_space="srgb", 
                              background_color=(dataset.get("metadata") or {}).get("background_color"))
        output = io.BytesIO()
        Image.fromarray(image).save(output, format="JPEG")
        output.seek(0)
        return send_file(output, mimetype="image/jpeg")

    del dataset_images

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

    app.config["TEMPLATES_AUTO_RELOAD"] = True
    return app


def run_viewer(model=None, train_dataset=None, test_dataset=None, nb_info=None):
    request_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    def render(feedid,
               pose=None,
               image_size=None,
               intrinsics=None,
               **kwargs):
        # image_size = np.array(image_size, dtype=np.int32)
        camera = nerfbaselines.new_cameras(
            poses=pose,
            intrinsics=np.array(intrinsics, dtype=np.float32),
            camera_models=0,
            image_sizes=image_size,
        )
        options = { "output_type_dtypes": { "color": "uint8" } }
                    #"embedding": embedding,
                   # "outputs": output_types }
        outputs = model.render(camera, options=options)
        frame = outputs["color"]
        return frame

    def update_step():
        req = request_queue.get()
        req_type = req.pop("type")
        if req_type == "render":
            feedid = req.pop("feedid")
            try:
                frame = render(feedid, **req)
                with io.BytesIO() as output, Image.fromarray(frame) as img:
                    img.save(output, format="JPEG")
                    output.seek(0)
                    frame_bytes = output.getvalue()
                output_queue.put({
                    "type": "result",
                    "feedid": feedid,
                    "result": frame_bytes,
                })
            except Exception as e:
                output_queue.put({
                    "type": "error",
                    "feedid": feedid,
                    "error": str(e),
                    "exception": e
                })

    process = multiprocessing.Process(target=_run_viewer_server, args=(
        request_queue, 
        output_queue, 
    ), kwargs=dict(
        datasets={"train": train_dataset, "test": test_dataset},
        dataset_metadata=train_dataset["metadata"],
    ), daemon=True)
    process.start()

    try:
        while True:
            update_step()
    finally:
        # Empty request queue
        while not request_queue.empty(): request_queue.pop()

        # Signal output queues to stop
        output_queue.put({ "type": "end" })
        process.kill()



if __name__ == "__main__":
    # run_viewer()
    from nerfbaselines import load_checkpoint
    from nerfbaselines.datasets import load_dataset

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, required=False, type=str)
    parser.add_argument("--data", type=str, default=None, required=False)
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--backend", type=str, default="python")
    args = parser.parse_args()

    port = 5001
    with ExitStack() as stack:
        nb_info = None
        model = None
        if args.checkpoint is not None:
            model, nb_info = stack.enter_context(load_checkpoint(args.checkpoint, backend=args.backend))
        else:
            logging.info("Starting viewer without method")

        train_dataset = None
        test_dataset = None
        if args.data is not None:
            train_dataset = load_dataset(args.data, split="train", load_features=False, features=("points3D_xyz", "points3D_rgb"))
            test_dataset = load_dataset(args.data, split="test", load_features=False, features=("points3D_xyz", "points3D_rgb"))

        # Start the viewer
        run_viewer(
            model=model, 
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            nb_info=nb_info)

