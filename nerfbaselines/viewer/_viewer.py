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


class _Feed:
    def __init__(self, feedid, render):
        self.feedid = feedid
        self.version = 0
        self._feed_params = {}
        self._state = {}
        self._render = render

    @property
    def feed_params(self):
        return self._feed_params

    @feed_params.setter
    def feed_params(self, value):
        self._feed_params = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.version += 1

    def render(self):
        return self._render(self.feedid, **copy.deepcopy(self.feed_params))


def _run_viewer_server(request_queue, output_queue, datasets):
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

    app = _build_flake_app(render_fn, datasets)
    multiplex_thread = threading.Thread(target=_multiplex_queues, args=(output_queue, output_queues), daemon=True)
    multiplex_thread.start()
    try:
        app.run(host="0.0.0.0", port=5001)
    finally:
        multiplex_thread.join()


def _build_flake_app(render_fn, datasets):
    app = Flask(__name__)
    _feed_states = {}

    @app.route("/")
    def index():
        return render_template("index.html")
    del index

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

    @app.route("/set-feed-params", methods=["POST"])
    def set_feed_params():
        feedid = request.args.get("feedid")
        if feedid is None:
            feedid = str(uuid.uuid4())
        if feedid not in _feed_states:
            _feed_states[feedid] = _Feed(feedid, render_fn)
        _feed_states[feedid].feed_params.update(request.json)
        return jsonify({"status": "ok", "feedid": feedid})
    del set_feed_params

    @app.route("/render")
    def render():
        req = request.args
        width = int(req.get("width", 256))
        height = int(req.get("height", 256))
        frame_bytes = render_fn(threading.get_ident(),
                                width=width, 
                                height=height)
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
        cameras = [{
            "pose": pose[:3, :4].flatten().tolist(),
            "intrinsics": intrinsics.tolist(),
            "image_size": image_size.tolist(),
        } for _, (pose, intrinsics, image_size) in enumerate(zip(
            nb_cameras.poses, 
            nb_cameras.intrinsics, 
            nb_cameras.image_sizes))]
        return jsonify({
            "cameras": cameras
        })
    del dataset_cameras

    @app.route("/dataset/thumbnails/<string:split>/<int:idx>.jpg")
    def dataset_thumbnail(split, idx):
        max_img_size = 64

        from nerfbaselines.datasets import dataset_index_select, dataset_load_features
        from nerfbaselines.utils import image_to_srgb

        if datasets.get(split) is None:
            return jsonify({"status": "error", "message": "Dataset not found"}), 404
        dataset = datasets[split]
        if not (0 <= idx < len(dataset["cameras"])): 
            return jsonify({"status": "error", "message": "Image not found in the dataset"}), 404

        dataset_slice = dataset_load_features(dataset_index_select(dataset, [idx]))
        image = dataset_slice["images"][0]

        W, H = image.shape[:2]
        downsample_factor = max(1, min(W//max_img_size, H//max_img_size))
        image = image[::downsample_factor, ::downsample_factor]
        image = image_to_srgb(image, 
                              dtype=np.uint8, 
                              color_space="srgb", 
                              background_color=(dataset.get("metadata") or {}).get("background_color"))
        output = io.BytesIO()
        Image.fromarray(image).save(output, format="JPEG")
        output.seek(0)
        return send_file(output, mimetype="image/jpeg")

    del dataset_thumbnail

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

        pose = nb_info["dataset_metadata"]["viewer_initial_pose"]
        image_size = np.array((256, 256), dtype=np.int32)
        if intrinsics is None:
            w, h = image_size
            intrinsics = np.array([w/2, w/2, w/2, h/2], dtype=np.float32)
        camera = nerfbaselines.new_cameras(
            poses=np.array(pose, dtype=np.float32),
            intrinsics=np.array(intrinsics, dtype=np.float32),
            camera_models=0,
            image_sizes=image_size,
        )
        outputs = model.render(camera)
        outputs = model.render(camera)
        frame = outputs["color"]
        return frame

    def update_step():
        req = request_queue.get()
        req_type = req.pop("type")
        if req_type == "render":
            feedid = req.pop("feedid")
            try:
                # frame = render(feedid, **req)

                frame = bouncing_ball_frame(time=time.time()*10, **req)
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

    process = multiprocessing.Process(target=_run_viewer_server, args=(request_queue, output_queue, {"train": train_dataset, "test": test_dataset}), daemon=True)
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

