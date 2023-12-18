import importlib
from threading import Thread
import types
from pathlib import Path
from time import sleep
import subprocess
import tempfile
import pickle
import base64
from typing import Optional, Tuple, Type, List, Dict
import os
import shutil
import hashlib
import traceback
import inspect
import random
import secrets
import logging
from dataclasses import dataclass, field, is_dataclass
from multiprocessing.connection import Listener, Client, Connection
from queue import Queue, Empty
from .types import Method, MethodInfo
from .types import NB_PREFIX  # noqa: F401
from .utils import partialmethod, cancellable, CancellationToken, CancelledException


PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ConnectionParams:
    port: int = field(default_factory=lambda: random.randint(10000, 20000))
    authkey: bytes = field(default_factory=lambda: secrets.token_hex(64).encode("ascii"))


def _report_ready():
    pass


def _remap_error(e: Exception):
    if e.__class__.__module__ == "builtins":
        return e
    elif e.__class__.__module__.startswith(_remap_error.__module__.split(".")[0]):
        return e

    # Remap exception
    return RuntimeError(f"Exception {e.__class__.__name__}: {e}")


def start_backend(method: Method, params: ConnectionParams, address: str = "localhost"):
    cancellation_token = CancellationToken()
    cancellation_tokens = {}

    input_queue = Queue(maxsize=3)
    output_queue = Queue(maxsize=32)

    def handler():
        with Listener((address, params.port), authkey=params.authkey) as listener:
            _report_ready()
            logging.info("Waiting for connection")
            with listener.accept() as conn:
                logging.info(f"Connection accepted from {listener.last_accepted}")
                while not conn.closed and not cancellation_token.cancelled:
                    if conn.poll():
                        msg = conn.recv()
                        message = msg["message"]
                        mid = msg["id"]

                        # do something with msg
                        if message == "close":
                            conn.send({"message": "close_ack", "id": mid})
                            cancellation_token.cancel()
                            break
                        if message == "cancel":
                            # if mid in cancellation_tokens:
                            conn.send({"message": "cancel_ack", "id": mid})
                            if mid in cancellation_tokens:
                                cancellation_tokens[mid].cancel()
                        elif message in {"call", "get"}:
                            if msg.get("cancellable", False):
                                cancellation_tokens[mid] = CancellationToken()
                            input_queue.put(msg)
                    elif not output_queue.empty():
                        conn.send(output_queue.get())
                    else:
                        sleep(0.0001)

    thread = Thread(target=handler, daemon=True)
    thread.start()

    while not cancellation_token.cancelled:
        try:
            msg = input_queue.get(timeout=0.1)
        except Empty:
            continue
        message = msg["message"]
        mid = msg["id"]

        if message == "get":
            logging.debug(f"Obtaining property {msg['property']}")
            try:
                result = getattr(method, msg["property"])
                if cancellation_token.cancelled:
                    break
                output_queue.put({"message": "result", "id": mid, "result": result})
            except Exception as e:  # pylint: disable=broad-except
                traceback.print_exc()
                logging.error(f"Error while obtaining property {msg['property']}")
                if cancellation_token.cancelled:
                    break
                output_queue.put({"message": "error", "id": mid, "error": _remap_error(e)})
        elif message == "call":
            try:
                method_or_fn = msg.get("function", msg.get("method"))
                if "function" in msg:
                    logging.debug(f"Calling function {msg['function']}")
                    splitter = msg["function"].rindex(".")
                    package, fnname = msg["function"][:splitter], msg["function"][splitter + 1 :]
                    fn = getattr(importlib.import_module(package), fnname)
                else:
                    logging.debug(f"Calling method {msg['method']}")
                    fn = getattr(method, msg["method"])
                kwargs = inject_callables(msg["kwargs"], output_queue, mid)
                args = inject_callables(msg["args"], output_queue, mid)
                if msg["cancellable"]:
                    fn = cancellable(fn)
                    kwargs["cancellation_token"] = cancellation_tokens[mid]
                result = fn(*args, **kwargs)
                if inspect.isgeneratorfunction(fn):
                    for r in result:
                        if cancellation_token.cancelled:
                            break
                        output_queue.put({"message": "yield", "id": mid, "yield": r})
                    result = None
                if cancellation_token.cancelled:
                    break
                output_queue.put({"message": "result", "id": mid, "result": result})
            except Exception as e:  # pylint: disable=broad-except
                if not isinstance(e, CancelledException):
                    traceback.print_exc()
                    logging.error(f"Error while calling method/function {method_or_fn} from")
                if cancellation_token.cancelled:
                    break
                output_queue.put({"message": "error", "id": mid, "error": _remap_error(e)})
            cancellation_tokens.pop(mid, None)
        else:
            logging.error(f"Unknown message {msg}")
            output_queue.put({"message": "error", "id": mid, "error": _remap_error(RuntimeError(f"Unknown message {msg}"))})
    logging.info("Client disconnected, shutting down")


class RemoteCallable:
    def __init__(self, i):
        self.id = i


def replace_callables(obj, callables, depth=0):
    if callable(obj):
        is_host = getattr(obj, "__host__", depth == 0)
        if is_host:
            callables.append(obj)
            return RemoteCallable(len(callables) - 1)
        else:
            return obj
    if isinstance(obj, dict):
        return {k: replace_callables(v, callables, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return obj.__class__((replace_callables(v, callables, depth + 1) for v in obj))
    if is_dataclass(obj):
        return obj.__class__(**{k: replace_callables(v, callables, depth + 1) for k, v in obj.__dict__.items()})
    return obj


def inject_callables(obj, output_queue, my_id):
    if isinstance(obj, RemoteCallable):

        def callback(*args, **kwargs):
            output_queue.put({"message": "callback", "id": my_id, "callback": obj.id, "args": args, "kwargs": kwargs})

        return callback
    if isinstance(obj, dict):
        return {k: inject_callables(v, output_queue, my_id) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return obj.__class__((inject_callables(v, output_queue, my_id) for v in obj))
    if is_dataclass(obj):
        return obj.__class__(**{k: inject_callables(v, output_queue, my_id) for k, v in obj.__dict__.items()})
    return obj


class RemoteMethod(Method):
    def __init__(self, *args, checkpoint: Optional[Path] = None, connection_params: Optional[ConnectionParams] = None, **kwargs):
        self.connection_params = connection_params or ConnectionParams()
        self._client: Optional[Connection] = None
        self._message_counter = 0
        self.args = args
        self.kwargs = kwargs
        self.checkpoint = checkpoint
        self._cancellation_tokens = {}

    @property
    def encoded_args(self):
        kwargs = self.kwargs
        if self.checkpoint is not None:
            checkpoint = self.checkpoint
            kwargs = dict(**self.kwargs, checkpoint=checkpoint)
        return base64.b64encode(pickle.dumps((self.args, kwargs))).decode("ascii")

    def get_info(self) -> MethodInfo:
        info = self._call("get_info")
        assert isinstance(info, MethodInfo), f"Invalid info type {type(info)}"
        return info

    @staticmethod
    def decode_args(encoded_args):
        return pickle.loads(base64.b64decode(encoded_args.encode("ascii")))

    @property
    def shared_path(self) -> Optional[Tuple[str, str]]:
        return None

    def _get_client(self):
        if self._client is None or self._client.closed:
            self._client = Client(("localhost", self.connection_params.port), authkey=self.connection_params.authkey)
        return self._client

    def _handle_call_result(self, client: Connection, my_id, callables):
        while not client.closed:
            if not client.poll():
                sleep(0.0001)
                continue
            message = client.recv()
            if message["id"] != my_id:
                continue
            elif message["message"] == "error":
                raise message["error"]
            elif message["message"] == "callback":
                callback = callables[message["callback"]]
                callback(*message["args"], **message["kwargs"])
                continue
            elif message["message"] == "cancel_ack":
                continue
            elif message["message"] == "result":
                return message["result"]
            elif "yield" in message:
                original_message = message

                def yield_fn():
                    yield original_message["yield"]
                    while not client.closed:
                        if not client.poll():
                            sleep(0.0001)
                            continue
                        message = client.recv()
                        if message["id"] != my_id:
                            continue
                        if message["message"] == "error":
                            raise message["error"]
                        if message["message"] == "callback":
                            callback = callables[message["callback"]]
                            callback(*message["args"], **message["kwargs"])
                            continue
                        if message["message"] == "result":
                            return
                        if message["message"] == "yield":
                            yield message["yield"]

                return yield_fn()
            else:
                raise RuntimeError(f"Unknown message {message}")
        raise RuntimeError("Connection closed")

    def _get(self, prop):
        client = self._get_client()
        mid = self._message_counter
        self._message_counter += 1
        callables: List[Dict] = []
        client.send({"message": "get", "id": mid, "property": prop})
        return self._handle_call_result(client, mid, callables)

    def _call(self, *args, cancellation_token: Optional[CancellationToken] = None, **kwargs):
        client = self._get_client()
        other_kwargs = {}
        if "function" in kwargs:
            assert "method" not in kwargs, "Cannot specify both method and function"
            other_kwargs["function"] = kwargs.pop("function")
        elif "method" in kwargs:
            other_kwargs["method"] = kwargs.pop("method")
        elif len(args) > 0:
            other_kwargs["method"] = args[0]
            args = args[1:]
        else:
            raise RuntimeError("Either method of function must be specified")
        mid = self._message_counter
        self._message_counter += 1
        callables: List[Dict] = []
        if cancellation_token is not None:
            assert type(cancellation_token).cancelled == CancellationToken.cancelled, "Custom cancellation tokens not supported"
            assert type(cancellation_token).cancel == CancellationToken.cancel, "Custom cancellation tokens not supported"
            old_cancel = cancellation_token.cancel

            def _call_cancel():
                client.send({"message": "cancel", "id": mid})
                old_cancel()

            cancellation_token.cancel = _call_cancel

        client.send(
            {
                "message": "call",
                "id": mid,
                "cancellable": cancellation_token is not None,
                "args": replace_callables(args, callables, depth=-1),
                "kwargs": replace_callables(kwargs, callables, depth=-1),
                **other_kwargs,
            }
        )
        return self._handle_call_result(client, mid, callables)

    def call(self, function: str, *args, **kwargs):
        return self._call(*args, function=function, **kwargs)

    @cancellable(mark_only=True)
    def train_iteration(self, *args, **kwargs):
        return self._call("train_iteration", *args, **kwargs)

    @cancellable(mark_only=True)
    def setup_train(self, *args, **kwargs):
        return self._call("setup_train", *args, **kwargs)

    @cancellable(mark_only=True)
    def render(self, *args, **kwargs):
        return self._call("render", *args, **kwargs)

    @cancellable(mark_only=True)
    def save(self, path: Path, **kwargs):
        if self.shared_path is not None:
            name = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
            local, remote = self.shared_path  # pylint: disable=unpacking-non-sequence
            local_path = os.path.join(local, name)
            remote_path = os.path.join(remote, name)
            shutil.copytree(str(path), local_path, dirs_exist_ok=True)
            self._call("save", Path(remote_path), **kwargs)
            shutil.copytree(local_path, str(path), dirs_exist_ok=True)
        else:
            self._call("save", path, **kwargs)

    def close(self):
        if self._client is not None:
            mid = self._message_counter + 1
            # Try recv
            try:
                self._client.send({"message": "close", "id": mid})
                while not self._client.closed:
                    try:
                        self._client.recv()
                    except EOFError:
                        break
            except BrokenPipeError:
                pass
            self._client.close()
            self._client = None


class RemoteProcessMethod(RemoteMethod):
    _local_address = "localhost"
    _package_path = PACKAGE_PATH
    build_code: Optional[str] = None
    python_path: str = "python"

    def __init__(self, *args, build_code: Optional[str] = None, python_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if build_code is not None:
            self.build_code = build_code
        if python_path is not None:
            self.python_path = python_path
        self._server_process: Optional[subprocess.Popen] = None
        self._tmp_shared_dir: Optional[tempfile.TemporaryDirectory] = None
        assert self.build_code is not None, "RemoteProcessMethod requires build_code to be specified"

    @classmethod
    def wrap(cls, method: Type[Method], **remote_kwargs):
        bases: Tuple[Type, ...] = (cls,)
        if issubclass(method, RemoteProcessMethod):
            bases = bases + (method,)
        elif "build_code" not in remote_kwargs:
            remote_kwargs["build_code"] = f"from {method.__module__} import {method.__name__}; method = {method.__name__}(*args, **kwargs)"

        def build(ns):
            ns["__module__"] = cls.__module__
            ns["__doc__"] = method.__doc__
            init_kwargs = {}
            for k, v in remote_kwargs.items():
                if k in method.__dict__ or k in cls.__dict__ or k in RemoteProcessMethod.__dict__:
                    ns[k] = v
                else:
                    init_kwargs[k] = v
            if init_kwargs:
                ns["__init__"] = partialmethod(cls.__init__, **init_kwargs)
            return ns

        return types.new_class(method.__name__, bases=bases, exec_body=build)

    def _get_server_process_args(self, env):
        if env.get("_NB_IS_DOCKERFILE", "0") == "1":
            return ["bash", "-l"]
        assert self._tmp_shared_dir is not None, "Temporary directory not created"
        is_verbose = logging.getLogger().isEnabledFor(logging.DEBUG)
        ready_path = self._tmp_shared_dir.name
        if self.shared_path is not None:
            ready_path = self.shared_path[1]  # pylint: disable=unsubscriptable-object
        code = f"""
import os, sys
sys.path.append(os.environ["NB_PATH"])
import nerfbaselines.communication
nerfbaselines.communication._report_ready = lambda: open("{ready_path}/ready", "w").close()
from nerfbaselines.utils import setup_logging

setup_logging(verbose={'True' if is_verbose else 'False'})

from nerfbaselines.communication import ConnectionParams, start_backend

args, kwargs = nerfbaselines.communication.RemoteMethod.decode_args(os.environ["NB_ARGS"])
{self.build_code or ''}
authkey = os.environ["NB_AUTHKEY"].encode("ascii")
start_backend(method, ConnectionParams(port=int(os.environ["NB_PORT"]), authkey=authkey), address="{self._local_address}")
"""
        return [self.python_path, "-c", code]

    @classmethod
    def _get_isolated_env(cls):
        safe_env = {
            "_NB_IS_DOCKERFILE",
            "PATH",
            "HOME",
            "USER",
            "LDLIBRARYPATH",
            "CXX",
            "CC",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "TCNN_CUDA_ARCHITECTURES",
            "TORCH_CUDA_ARCH_LIST",
            "CUDAARCHS",
            "GITHUB_ACTIONS",
            "CONDA_PKGS_DIRS",
            "PIP_CACHE_DIR",
            "TORCH_HOME",
        }
        return {k: v for k, v in os.environ.items() if k.upper() in safe_env or k.startswith("NB_")}

    def _ensure_server_running(self):
        if self._server_process is None:
            self._tmp_shared_dir = tempfile.TemporaryDirectory()
            env = self._get_isolated_env()
            env["NB_PORT"] = str(self.connection_params.port)
            env["NB_PATH"] = self._package_path
            env["NB_AUTHKEY"] = self.connection_params.authkey.decode("ascii")
            env["NB_ARGS"] = self.encoded_args
            args = self._get_server_process_args(env)
            self._server_process = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL)
            ready_path = self._tmp_shared_dir.name
            if self.shared_path is not None:
                ready_path = self.shared_path[0]  # pylint: disable=unsubscriptable-object
            while not os.path.exists(os.path.join(ready_path, "ready")) and self._server_process.poll() is None:
                sleep(1)
            if self._server_process.poll() is not None:
                raise RuntimeError("Server died")
            logging.info("Server started")

    def _get_client(self):
        self._ensure_server_running()
        return super()._get_client()

    @classmethod
    def _get_install_args(cls) -> Optional[List[str]]:
        return None

    @classmethod
    def install(cls):
        args = cls._get_install_args()  # pylint: disable=assignment-from-none
        if args is not None:
            subprocess.check_call(args, env=cls._get_isolated_env())

    def close(self):
        super().close()
        if self._server_process is not None:
            if self._server_process.poll() is None:
                try:
                    self._server_process.wait(6)
                except subprocess.TimeoutExpired:
                    pass
            if self._server_process.poll() is None:
                logging.info("Waiting for the server to shut down")
                try:
                    self._server_process.wait(10)
                except subprocess.TimeoutExpired:
                    pass
            if self._server_process.poll() is None:
                logging.error("Server did not shut down, killing")
                self._server_process.kill()
                self._server_process.wait()
        if self._tmp_shared_dir is not None:
            self._tmp_shared_dir.cleanup()
