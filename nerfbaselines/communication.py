from functools import cached_property
import types
from time import sleep
import subprocess
import tempfile
import pickle
import base64
from typing import Optional, Tuple, Type
import os
import shutil
import hashlib
import traceback
import inspect
import random
import secrets
import logging
from dataclasses import dataclass, field, is_dataclass
from multiprocessing.connection import Listener, Client
from .types import Method, MethodInfo
from .utils import partialmethod


PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PREFIX = os.path.expanduser(os.environ.get("NB_PREFIX", "~/.cache/nerfbaselines"))


@dataclass
class ConnectionParams:
    port: int = field(default_factory=lambda: random.randint(10000, 20000))
    authkey: bytes = field(default_factory=lambda: secrets.token_hex(64).encode("ascii"))


def _report_ready():
    pass


def start_backend(method: Method, params: ConnectionParams, address: str = "localhost"):
    with Listener((address, params.port), authkey=params.authkey) as listener:
        _report_ready()
        logging.info("Waiting for connection")
        with listener.accept() as conn:
            logging.info(f"Connection accepted from {listener.last_accepted}")
            while not conn.closed:
                msg = conn.recv()
                message = msg["message"]
                mid = msg["id"]
                # do something with msg
                if message == 'close':
                    conn.send({"message": "close_ack"})
                    break
                elif message == 'get':
                    logging.debug(f"Obtaining property {msg['property']} from {listener.last_accepted}")
                    try:
                        result = getattr(method, msg["property"])
                        conn.send({"message": "result", "id": mid, "result": result})
                    except Exception as e:  # pylint: disable=broad-except
                        traceback.print_exc()
                        logging.error(f"Error while obtaining property {msg['property']} from {listener.last_accepted}")
                        conn.send({"message": "error", "id": mid, "error": e})
                elif message == 'call':
                    logging.debug(f"Calling method {msg['method']} from {listener.last_accepted}")
                    try:
                        fn = getattr(method, msg["method"])
                        result = fn(*inject_callables(msg["args"], conn, mid), **inject_callables(msg["kwargs"], conn, mid))
                        if inspect.isgeneratorfunction(fn):
                            for r in enumerate(result):
                                conn.send({"message": "yield", "id": mid, "yield": r})
                            result = None
                        conn.send({"message": "result", "id": mid, "result": result})
                    except Exception as e:  # pylint: disable=broad-except
                        traceback.print_exc()
                        logging.error(f"Error while calling method {msg['method']} from {listener.last_accepted}")
                        conn.send({"message": "error", "id": mid, "error": e})
                else:
                    logging.error(f"Unknown message {msg} from {listener.last_accepted}")
                    conn.send({"message": "error", "id": mid, "error": RuntimeError(f"Unknown message {msg}")})
        logging.info("Client disconnected, shutting down")


class RemoteCallable:
    def __init__(self, i):
        self.id = i


def replace_callables(obj, callables):
    if callable(obj):
        callables.append(obj)
        return RemoteCallable(len(callables)-1)
    if isinstance(obj, dict):
        return {k: replace_callables(v, callables) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return obj.__class__((replace_callables(v, callables) for v in obj))
    if is_dataclass(obj):
        return obj.__class__(**{k: replace_callables(v, callables) for k, v in obj.__dict__.items()})
    return obj


def inject_callables(obj, conn, my_id):
    if isinstance(obj, RemoteCallable):
        def callback(*args, **kwargs):
            conn.send({
                "message": "callback",
                "id":my_id,
                "callback": obj.id,
                "args": args,
                "kwargs": kwargs})
        return callback
    if isinstance(obj, dict):
        return {k: inject_callables(v, conn, my_id) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return obj.__class__((inject_callables(v, conn, my_id) for v in obj))
    if is_dataclass(obj):
        return obj.__class__(**{k: inject_callables(v, conn, my_id) for k, v in obj.__dict__.items()})
    return obj


class RemoteMethod(Method):
    def __init__(self, *args, checkpoint: Optional[str] = None, connection_params: Optional[ConnectionParams] = None, **kwargs):
        self.connection_params = connection_params or ConnectionParams()
        self._client = None
        self._message_counter = 0
        self.args = args
        self.kwargs = kwargs
        self.checkpoint = checkpoint

    @property
    def encoded_args(self):
        kwargs = self.kwargs
        if self.checkpoint is not None:
            checkpoint = self.checkpoint
            kwargs = dict(**self.kwargs, checkpoint=checkpoint)
        return base64.b64encode(pickle.dumps((self.args, kwargs))).decode("ascii")

    @cached_property
    def info(self) -> MethodInfo:
        return self._get("info")

    @staticmethod
    def decode_args(encoded_args):
        return pickle.loads(base64.b64decode(encoded_args.encode("ascii")))

    @property
    def shared_path(self) -> Optional[Tuple[str, str]]:
        return None

    def _get_client(self):
        if self._client is None or self._client.closed:
            self._client = Client(('localhost', self.connection_params.port), authkey=self.connection_params.authkey)
        return self._client

    def _handle_call_result(self, client: Client, my_id, callables):
        while not client.closed:
            message = client.recv()
            if message["id"] != my_id:
                continue
            elif message["message"] == "error":
                raise message["error"]
            elif message["message"] == "callback":
                callback = callables[message["callback"]]
                callback(*message["args"], **message["kwargs"])
                continue
            elif message["message"] == "result":
                return message["result"]
            elif "yield" in message:
                original_message = message
                def yield_fn():
                    yield original_message["yield"]
                    while not client.closed:
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
        callables = []
        client.send({
            "message": "get",
            "id": mid,
            "property": prop})
        return self._handle_call_result(client, mid, callables)


    def _call(self, method, *args, **kwargs):
        client = self._get_client()
        mid = self._message_counter
        self._message_counter += 1
        callables = []
        client.send({
            "message": "call",
            "id": mid,
            "method": method,
            "args": replace_callables(args, callables),
            "kwargs": replace_callables(kwargs, callables)})
        return self._handle_call_result(client, mid, callables)

    def train_iteration(self, *args, **kwargs):
        return self._call("train_iteration", *args, **kwargs)

    def setup_train(self, *args, **kwargs):
        return self._call("setup_train", *args, **kwargs)

    def render(self, *args, **kwargs):
        return self._call("render", *args, **kwargs)

    def save(self, path: str):
        if self.shared_path is not None:
            name = hashlib.sha256(path.encode("utf-8")).hexdigest()
            local, remote = self.shared_path  # pylint: disable=unpacking-non-sequence
            local_path = os.path.join(local, name)
            remote_path = os.path.join(remote, name)
            shutil.copytree(path, local_path, dirs_exist_ok=True)
            self._call("save", remote_path)
            shutil.copytree(local_path, path, dirs_exist_ok=True)
        else:
            self._call("save", path)

    def close(self):
        if self._client is not None:
            mid = self._message_counter+1
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
    build_code: str = None
    python_path: str = "python"

    def __init__(self, *args, build_code: Optional[str] = None, python_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if build_code is not None:
            self.build_code = build_code
        if python_path is not None:
            self.python_path = python_path
        self._server_process = None
        self._tmp_shared_dir = None
        assert self.build_code is not None, "RemoteProcessMethod requires build_code to be specified"

    @classmethod
    def wrap(cls, method: Type[Method], **remote_kwargs):
        bases = (cls,)
        if issubclass(method, RemoteProcessMethod):
            bases = bases + (method,)
        else:
            if "build_code" not in remote_kwargs:
                remote_kwargs["build_code"] = f"from {method.__module__} import {method.__name__}; method = {method.__name__}(*args, **kwargs)"
        def build(ns):
            ns["__module__"] = cls.__module__
            ns["__doc__"] = method.__doc__
            if remote_kwargs:
                ns["__init__"] = partialmethod(cls.__init__, **remote_kwargs)
            for k, v in remote_kwargs.items():
                if k in method.__dict__ or k in cls.__dict__:
                    ns[k] = v
            return ns
        return types.new_class(method.__name__, bases=bases, exec_body=build)

    def _get_server_process_args(self, env):
        if env.get("_NB_IS_DOCKERFILE", "0") == "1":
            return ["bash", "-l"]
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

    def _get_isolated_env(self):
        safe_env = {
            "PATH", "HOME", "USER", "LDLIBRARYPATH",
             "CXX", "CC", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"}
        return {k: v for k, v in os.environ.items() if k.upper() in safe_env or k.startswith("NB_")}

    def _ensure_server_running(self):
        if self._server_process is None:
            self._tmp_shared_dir = tempfile.TemporaryDirectory()
            env = self._get_isolated_env()
            env["NB_PORT"] = str(self.connection_params.port)
            env["NB_PATH"] = PACKAGE_PATH
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

    def _get_install_args(self):
        return None

    def install(self):
        args = self._get_install_args()  # pylint: disable=assignment-from-none
        if args is not None:
            subprocess.check_call(args, env=self._get_isolated_env())

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