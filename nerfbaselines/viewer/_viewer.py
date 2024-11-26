import multiprocessing
import threading
import copy
import queue
import time
import logging
import uuid
import io
from flask import Flask, request, jsonify, render_template, Response
import numpy as np
from PIL import Image, ImageDraw

pcs = set()
candidates = []

TEMPLATES_AUTO_RELOAD = True


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


def _run_viewer_server(request_queue, output_queue):
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
            raise RuntimeError(out["message"]) from out["exception"]
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
            logging.error(e)

    app = _build_flake_app(render_fn)
    multiplex_thread = threading.Thread(target=_multiplex_queues, args=(output_queue, output_queues), daemon=True)
    multiplex_thread.start()
    try:
        app.run(host="0.0.0.0", port=5001)
    finally:
        multiplex_thread.join()


def _build_flake_app(render_fn):
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


def run_viewer():
    request_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    def update_step():
        req = request_queue.get()
        req_type = req.pop("type")
        if req_type == "render":
            feedid = req.pop("feedid")
            try:
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

    process = multiprocessing.Process(target=_run_viewer_server, args=(request_queue, output_queue), daemon=True)
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
    run_viewer()
