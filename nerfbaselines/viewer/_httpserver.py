"""
In this file, we implement two different implementations of the HTTP server for the viewer.
The first implementation is a simple HTTP server using the built-in Python HTTP server.
The second implementation is a Flask server that provides a more feature-rich experience.
"""
import errno
import socket
import shutil
from functools import partial
from typing import cast, Any
import queue
import contextlib
import logging
import os
import struct
import json
import io
import threading
import numpy as np
from PIL import Image
from nerfbaselines.datasets import dataset_index_select, dataset_load_features
from nerfbaselines.utils import image_to_srgb
from ._proxy import cloudflared_tunnel
from ._websocket import httpserver_websocket_handler, ConnectionClosed
from ._static import get_palettes_js
from nerfbaselines.backends._common import setup_logging as setup_logging


logger = logging.getLogger("nerfbaselines.viewer")


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


@contextlib.contextmanager
def _make_render_fn(request_queue, output_queue):
    output_queues = {}
    message_counter = threading.local()

    def render_fn(feedid, **kwargs):
        mid = getattr(message_counter, "counter", 0)
        message_counter.counter = mid + 1
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
                    break
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
        output_queue.put({"type": "end"})
        multiplex_thread.join()


def write_dataset_pointcloud(file, points3D_xyz, points3D_rgb=None):
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
    plydata.write(file)


class ViewerBackend:
    def __init__(self, request_queue, output_queue, *, info, datasets):
        self._info = info
        self._datasets = datasets
        self._images_cache = {}
        self._render_fn = None
        self._stack = contextlib.ExitStack()
        self._request_queue = request_queue
        self._output_queue = output_queue
        self._thumbnail_size = 96

    def __enter__(self):
        self._stack.__enter__()
        self._render_fn = self._stack.enter_context(
            _make_render_fn(self._request_queue, self._output_queue))
        return self

    def __exit__(self, *args):
        self._stack.__exit__(*args)
        del args

    def notify_started(self, port):
        self._request_queue.put({
            "type": "start",
            "port": port,
            "thread_id": 0,
        })

    def get_info(self):
        return self._info

    def render(self, req):
        assert self._render_fn is not None, "Backend not initialized"
        if "image_size" not in req:
            raise ValueError("Invalid request, missing image_size")
        if "pose" not in req:
            raise ValueError("Invalid request, missing pose")
        if "intrinsics" not in req:
            raise ValueError("Invalid request, missing intrinsics")
        if "output_type" not in req:
            raise ValueError("Invalid request, missing output_type")
        image_size = [int(x) for x in req["image_size"]]
        pose = np.array(req.get("pose"), dtype=np.float32).reshape(3, 4)
        intrinsics = np.array(req.get("intrinsics"), dtype=np.float32)
        lossless = req.get("lossless", False)
        frame_bytes = self._render_fn(
            threading.get_ident(), 
            pose=pose, 
            image_size=image_size, 
            intrinsics=intrinsics,
            output_type=req.get("output_type"),
            palette=req.get("palette") or None,
            output_range=req.get("output_range") or (None, None),
            appearance_weights=req.get("appearance_weights"),
            appearance_train_indices=req.get("appearance_train_indices"),
            split_percentage=float(req.get("split_percentage", 0.5)),
            split_palette=req.get("split_palette") or None,
            split_tilt=float(req.get("split_tilt", 0)),
            format="PNG" if lossless else "JPEG",
            split_range=req.get("split_range") or (None, None),
            split_output_type=req.get("split_output_type"))
        mimetype = "image/png" if lossless else "image/jpeg"
        return frame_bytes, mimetype

    def get_dataset_image(self, req):
        if "split" not in req:
            raise ValueError("Invalid request, missing split")
        if "idx" not in req:
            raise ValueError("Invalid request, missing idx")
        idx = int(req["idx"])
        split = req["split"]
        max_img_size = None
        if req.get("thumb") == "1":
            max_img_size = self._thumbnail_size
        if (split, idx, max_img_size) in self._images_cache:
            out = self._images_cache[(split, idx, max_img_size)]
            return out

        if self._datasets.get(split) is None:
            raise NotFound("Dataset not found")
        dataset = self._datasets[split]
        if not (0 <= idx < len(dataset["cameras"])): 
            raise NotFound("Image not found in the dataset")

        # Disable logging
        logger = logging.getLogger("nerfbaselines.datasets")
        save_level = logger.level
        try:
            logger.setLevel(logging.ERROR)
            dataset_slice = dataset_load_features(dataset_index_select(dataset, [idx]), show_progress=False)
        finally:
            logger.setLevel(save_level)
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
            self._images_cache[(split, idx, max_img_size)] = out, "image/jpeg"
        return out, "image/jpeg"

    def create_public_url(self, port, accept_license_terms):
        if not accept_license_terms:
            return {
                "status": "error",
                "message": "License terms (https://www.cloudflare.com/website-terms/) must be accepted",
                "license_terms_url": "https://www.cloudflare.com/website-terms/"
            }
        local_url = f"http://localhost:{port}"
        public_url = self._stack.enter_context(cloudflared_tunnel(local_url, accept_license_terms=accept_license_terms))
        self._info["state"]["viewer_public_url"] = public_url
        return {"status": "ok", "public_url": public_url}

    def handle_render_websocket_message(self, data):
        thread = None
        try:
            assert isinstance(data, str)
            req = json.loads(data)
            thread = req.pop("thread")
            payload, mimetype = self.render(req)
            msg_bytes = json.dumps({
                "status": "ok", "thread": thread, "mimetype": mimetype
            }).encode("utf-8")
            message_length = len(msg_bytes)
            return struct.pack(f"!I", message_length) + msg_bytes + payload
        except Exception as e:
            logging.exception(e)
            return json.dumps({
                "status": "error", "message": str(e), "thread": thread
            })

    def get_dataset_pointcloud(self):
        dataset = self._datasets.get("train")
        if dataset is None:
            raise NotFound("Dataset not found")
        points3D_xyz = dataset.get("points3D_xyz")
        points3D_rgb = dataset.get("points3D_rgb")
        if points3D_xyz is None:
            raise NotFound("No pointcloud in dataset")
        output = io.BytesIO()
        write_dataset_pointcloud(output, points3D_xyz, points3D_rgb)
        output.seek(0)
        return output

    def get_dataset_split_cameras(self, split, get_image_url, get_thumbnail_url):
        if not isinstance(split, str):
            raise ValueError("Invalid split specified")
        dataset = self._datasets.get(split)
        if dataset is None:
            raise NotFound("Dataset not found")
        nb_cameras = dataset.get("cameras")
        root_path = dataset.get("image_paths_root")
        if root_path is None:
            root_path = os.path.commonpath(dataset["image_paths"])
        cameras = [{
            "pose": pose[:3, :4].flatten().tolist(),
            "intrinsics": intrinsics.tolist(),
            "image_size": image_size.tolist(),
            "image_name": os.path.relpath(image_path, root_path),
            "image_url": get_image_url(split, i, image_path) if get_image_url is not None else None,
            "thumbnail_url": get_thumbnail_url(split, i, image_path),
        } for i, (pose, intrinsics, image_size, image_path) in enumerate(zip(
            nb_cameras.poses, 
            nb_cameras.intrinsics, 
            nb_cameras.image_sizes,
            dataset["image_paths"]))]
        return { "cameras": cameras }

    def get_dataset(self, get_image_url, get_thumbnail_url):
        return {
            "train": self.get_dataset_split_cameras("train", get_image_url, get_thumbnail_url) if self._datasets.get("train") is not None else None,
            "test": self.get_dataset_split_cameras("test", get_image_url, get_thumbnail_url) if self._datasets.get("test") is not None else None,
        }


from http.server import SimpleHTTPRequestHandler


def httpserver_json_errorhandler(fn):
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            out_data = {"status": "error", "message": str(e)}
            status = 500
            if isinstance(e, BadRequest):
                status = 400
            elif isinstance(e, NotFound):
                status = 404
            else:
                # We want to log all other exceptions
                logging.exception(e)
            self.send_response(status)
            out = json.dumps(out_data).encode("utf-8")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
    return wrapper


class ViewerRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, request, client_address, server_class, backend):
        self.backend = backend
        self.server_class = server_class
        static_directory = os.path.join(os.path.dirname(__file__), "static")
        self._templates = {}
        templates_directory = os.path.join(os.path.dirname(__file__), "static")
        with open(os.path.join(templates_directory, "index.html"), "r") as f:
            self._templates["index.html"] = f.read()
        super().__init__(request, client_address, server_class, directory=static_directory)

    def log_message(self, format, *args):
        if logger.isEnabledFor(logging.DEBUG):
            message = format % args
            logger.debug("%s - - [%s] %s" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          message.translate(cast(Any, self)._control_char_table)))

    def list_directory(self, path):
        # Disabling directory listing
        del path
        self.send_error(404, "Not Found")
        return None

    @httpserver_json_errorhandler
    def _handle_index(self):
        html = self._templates["index.html"].replace("{{ data|safe }}", json.dumps(self.backend.get_info()))
        content = html.encode("utf-8")
        f = io.BytesIO()
        f.write(content)
        f.seek(0)

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cross-Origin-Opener-Policy", "same-origin");
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp");
        self.end_headers()
        return f

    @httpserver_json_errorhandler
    def _handle_palettes_js(self):
        output_bytes = get_palettes_js().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-type", "application/javascript")
        self.send_header("Content-Length", str(len(output_bytes)))
        self.end_headers()
        self.wfile.write(output_bytes)

    @httpserver_json_errorhandler
    def _handle_dataset_cameras(self):
        output = self.backend.get_dataset(
            lambda split, idx, _: f"./dataset/images/{split}/{idx}.jpg",
            lambda split, idx, _: f"./dataset/images/{split}/{idx}.jpg?thumb=1")
        output_bytes = json.dumps(output).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(output_bytes)))
        self.end_headers()
        self.wfile.write(output_bytes)

    @httpserver_json_errorhandler
    def _handle_dataset_pointcloud(self):
        with self.backend.get_dataset_pointcloud() as output:
            self.send_response(200)
            self.send_header("Content-type", "application/octet-stream")
            self.send_header("Content-Length", str(output.getbuffer().nbytes))
            self.end_headers()
            shutil.copyfileobj(output, self.wfile)

    def _get_query(self):
        query = {}
        if "?" not in self.path:
            return query
        query_string = self.path.split("?", 1)[-1]
        for part in query_string.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                query[k] = v
        return query

    def send_header(self, keyword, value):
        if keyword == "Content-type" and value == "text/javascript":
            super().send_header("Cross-Origin-Opener-Policy", "same-origin")
            super().send_header("Cross-Origin-Embedder-Policy", "require-corp")
        return super().send_header(keyword, value)

    @httpserver_json_errorhandler
    def _handle_dataset_image(self, split, idx):
        query = self._get_query()
        req = {"thumb": query.get("thumb"), "split": split, "idx": idx}
        data, mimetype = self.backend.get_dataset_image(req)
        self.send_response(200)
        self.send_header("Content-type", mimetype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @httpserver_json_errorhandler
    def _handle_create_public_url(self):
        query = self._get_query()
        accept_license_terms = query.get("accept_license_terms") == "yes"
        port = self.server_class.server_address[1]
        output = self.backend.create_public_url(port, accept_license_terms)
        output_bytes = json.dumps(output).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(output_bytes)))
        self.end_headers()
        self.wfile.write(output_bytes)

    @httpserver_json_errorhandler
    def _handle_render_http(self):
        # Read input json
        try:
            content_length = int(self.headers.get("content-length", 0))
            req_bytes = self.rfile.read(content_length)
            req = json.loads(req_bytes)
        except Exception as e:
            logging.exception(e)
            raise BadRequest("Invalid JSON request")
        payload, mimetype = self.backend.render(req)
        self.send_response(200)
        self.send_header("Content-type", mimetype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @httpserver_websocket_handler
    def _handle_render_websocket(self, ws):
        try:
            while True:
                reqdata = ws.receive()
                out = self.backend.handle_render_websocket_message(reqdata)
                ws.send(out)
        except ConnectionClosed:
            pass
        except Exception as e:
            logging.exception(e)

    def send_head(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path == "/":
            return self._handle_index()
        return super().send_head()

    def do_GET(self):
        path = self.path
        if "?" in path:
            path = self.path.split("?", 1)[0]
        if path == "/render-websocket":
            return self._handle_render_websocket()
        if path == "/dataset.json":
            return self._handle_dataset_cameras()
        if path == "/palettes.js":
            return self._handle_palettes_js()
        if path == "/dataset/pointcloud.ply":
            return self._handle_dataset_pointcloud()
        if path.startswith("/dataset/images/") and path.endswith(".jpg"):
            split = path[len("/dataset/images/"): -len(".jpg")]
            if "/" in split:
                split, idx = split.split("/", 1)
                try:
                    idx = int(idx)
                    return self._handle_dataset_image(split, idx)
                except Exception:
                    pass
        return super().do_GET()

    def do_POST(self):
        path = self.path
        if "?" in path:
            path = self.path.split("?", 1)[0]
        if path == "/create-public-url":
            return self._handle_create_public_url()
        if path == "/render":
            return self._handle_render_http()
        return self.send_error(404)


def run_simple_http_server(*args, host=None, port=None, verbose=False, **kwargs):
    if host is None or host == "" or host == "localhost":
        host = ""
    setup_logging(verbose=verbose)
    from http.server import ThreadingHTTPServer
    if port is None:
        port = 0

    class ThreadingHTTPServerWithBind(ThreadingHTTPServer):
        def server_bind(self):
            host, port = self.server_address
            for _ in range(100):
                try:
                    out = super().server_bind()
                    break
                except OSError as e:
                    if e.errno == errno.EADDRINUSE and port > 0:
                        port = port + 1
                        self.server_address = (host, port)
                        continue
                    raise
            else:
                raise RuntimeError("Could not find a free port")
            port = self.server_address[1]
            backend.notify_started(port)
            return out

    try:
        if ":" in host:
            ThreadingHTTPServer.address_family = socket.AF_INET6
        with ViewerBackend(*args, **kwargs) as backend, \
                ThreadingHTTPServerWithBind((host, port), partial(ViewerRequestHandler, backend=backend)) as server:
            port = server.server_address[1]
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.exception(e)


def run_flask_server(*args, port, host=None, verbose=False, **kwargs):
    if host is None or host == "" or host == "localhost":
        host = "127.0.0.1"
    setup_logging(verbose=verbose)
    from flask import Flask, request, jsonify, render_template, Response, send_file
    from flask import request, Response, send_from_directory, make_response
    import socketserver
    from ._websocket import flask_websocket_route

    # Create app
    app = Flask(__name__, static_url_path="")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    # Reduce logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.DEBUG if verbose else logging.ERROR)

    # Full exception details
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.error(f"Server Error: {e}", exc_info=True)
        if isinstance(e, BadRequest):
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 400
        if isinstance(e, NotFound):
            return jsonify({
                "status": "error",
                "message": str(e),
            }), 404
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500
    del handle_exception

    @app.errorhandler(500)
    def handle_500_error(e):
        app.logger.error(f"Server Error: {e}", exc_info=True)  # Log the error with traceback
        return jsonify({
            "status": "error",
            "error": "Internal Server Error",
        }), 500
    del handle_500_error

    @app.errorhandler(404)
    def handle_404_error(e):
        del e
        app.logger.error(f"404 ERROR (Path not found): {request.path}")
        return jsonify({
            "status": "error",
            "error": f"Path not found: {request.path}",
        }), 404
    del handle_404_error

    # Add required headers for .js files
    @app.after_request
    def add_header(response):
        response.headers["cross-origin-opener-policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        return response
    del add_header

    @app.route("/")
    def index():
        response = make_response(render_template("index.html", data=json.dumps(backend.get_info())))
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        return response
    del index

    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(os.path.join(app.root_path, 'static'),
                                   'favicon.ico', mimetype='image/vnd.microsoft.icon')
    del favicon

    @flask_websocket_route(app, "/render-websocket")
    def render_websocket(ws):
        try:
            while True:
                reqdata = ws.receive()
                out = backend.handle_render_websocket_message(reqdata)
                ws.send(out)
        except ConnectionClosed:
            pass
        except Exception as e:
            logging.exception(e)
    del render_websocket

    @app.route("/render", methods=["POST"])
    def render():
        req = request.json
        payload, mimetype = backend.render(req)
        return Response(payload, mimetype=mimetype)
    del render

    @app.route("/palettes.js")
    def palettes_js():
        return Response(get_palettes_js(), mimetype="application/javascript")
    del palettes_js

    @app.route("/create-public-url", methods=["POST"])
    def create_public_url():
        # accept_license_terms is a query param
        accept_license_terms = request.args.get("accept_license_terms") == "yes"
        output = backend.create_public_url(port, accept_license_terms)
        return jsonify(output)
    del create_public_url

    @app.route("/dataset/pointcloud.ply")
    def dataset_pointcloud():
        output = backend.get_dataset_pointcloud()
        return send_file(output, download_name="pointcloud.ply", as_attachment=True)
    del dataset_pointcloud

    @app.route("/dataset.json")
    def dataset_cameras():
        output = backend.get_dataset(
            lambda split, idx, _: f"./dataset/images/{split}/{idx}.jpg",
            lambda split, idx, _: f"./dataset/images/{split}/{idx}.jpg?thumb=1")
        return jsonify(output)
    del dataset_cameras

    @app.route("/dataset/images/<string:split>/<int:idx>.jpg")
    def _get_dataset_image_route(split, idx):
        out, mimetype = backend.get_dataset_image({
            "split": split, 
            "idx": idx, 
            "thumb": request.args.get("thumb")})
        return Response(out, mimetype=mimetype)
    del _get_dataset_image_route

    original_socket_bind = socketserver.TCPServer.server_bind
    def socket_bind_wrapper(self):
        nonlocal port
        try:
            for _ in range(100):
                try:
                    ret = original_socket_bind(self)
                    _, port = self.socket.getsockname()
                    break
                except OSError as e:
                    if e.errno == errno.EADDRINUSE and port > 0:
                        port = port + 1
                        continue
                    raise
            else:
                raise RuntimeError("Could not find a free port")
            backend.notify_started(port)
            # Recover original implementation
            return ret
        finally:
            socketserver.TCPServer.server_bind = original_socket_bind

    try:
        with ViewerBackend(*args, **kwargs) as backend:
            try:
                socketserver.TCPServer.server_bind = socket_bind_wrapper   #Hook the wrapper
                app.run(host=host, port=port or 0)
            finally:
                socketserver.TCPServer.server_bind = original_socket_bind
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.exception(e)
