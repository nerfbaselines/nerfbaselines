import contextlib
from itertools import chain
import traceback
import sys
import subprocess
from pathlib import Path
import os
import random
import dataclasses
from functools import partial
import types
from dataclasses import dataclass, is_dataclass
import time
import importlib
from threading import Event
from time import sleep
import pickle
import socket
from typing import Optional, List, Any, Dict, Callable, cast, Tuple
import inspect
import secrets
import logging
from multiprocessing.connection import Listener, Client, Connection
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from ..utils import cancellable, CancellationToken, CancelledException
from ._common import Backend
import nerfbaselines


_MESSAGE_COUNTER = 0
MESSAGE_SIZE = 128 * 1024 * 1024  # 128 MB


def _remap_error(e: Exception):
    if e.__class__.__module__ == "builtins":
        return e
    elif e.__class__.__module__.startswith(_remap_error.__module__.split(".")[0]):
        return e

    # Remap exception
    return RuntimeError(f"Exception {e.__class__.__name__}: {e}")


def send(connection: Connection, message):
    # msgo = message.copy()
    # msgo.pop("kwargs", None)
    # msgo.pop("args", None)
    # print("send", msgo)
    message_bytes = pickle.dumps(message)
    for i in range(0, len(message_bytes), MESSAGE_SIZE):
        connection.send_bytes(message_bytes[i : i + MESSAGE_SIZE])
    if len(message_bytes) % MESSAGE_SIZE == 0:
        connection.send_bytes(b"")

def build_recv(connection: Connection):
    def recv():
        message_bytes = connection.recv_bytes()
        message_len = len(message_bytes)
        while message_len == MESSAGE_SIZE:
            new_message = connection.recv_bytes()
            message_len = len(new_message)
            message_bytes += new_message
        # msgo = pickle.loads(message_bytes)
        # msgo.pop("kwargs", None)
        # msgo.pop("args", None)
        # print("recv", msgo)
        return pickle.loads(message_bytes)
    return recv


@dataclass(eq=True, frozen=True)
class VirtualInstance:
    id: int
    methods: List[str]
    attrs: List[str]

    @staticmethod
    def get_virtual_instance(obj):
        obj_cls = obj
        methods = []
        attrs = []
        ignore_members = {
            "__new__", "__init__", 
            "__getattribute__", "__getattr__", "__setattr__"}
        obj_cls = obj
        members = {}
        members.update({k:v for k, v in inspect.getmembers(obj_cls) if k not in ignore_members})
        if not isinstance(obj, type):
            obj_cls = type(obj)
            members.update({k:v for k, v in inspect.getmembers(obj_cls) if k not in ignore_members})
        for k, v in members.items():
            if isinstance(v, (classmethod, staticmethod)):
                methods.append(k)
            elif callable(v):
                methods.append(k)
            elif isinstance(v, property):
                attrs.append(k)
        return VirtualInstance(id(obj), 
                               methods=methods, 
                               attrs=attrs)

    def build_wrapper(self, backend: Backend):
        instance_id = self.id
        ns = {}
        class classproperty(object):
            def __init__(self, instance_id, k):
                self.instance_id = instance_id
                self.k = k
            def __get__(self, owner_self, owner_cls):
                return backend.instance_getattr(self.instance_id, self.k)
        for k in self.methods:
            ns[k] = staticmethod(partial(backend.instance_call, instance_id, k))
        for k in self.attrs:
            ns[k] = classproperty(instance_id, k)
        ns["__call__"] = staticmethod(partial(backend.instance_call, instance_id, "__call__"))
        ns["__repr__"] = lambda x: f"<VirtualInstance {instance_id}>"
        ns["__str__"] = lambda x: f"<VirtualInstance {instance_id}>"
        ns["__del__"] = staticmethod(partial(backend.instance_del, instance_id))
        return types.new_class("VirtualInstanceRPC", (), {}, exec_body=lambda _ns: _ns.update(ns))()


def replace_instances(registry, obj):
    a = registry,
    if obj is None:
        return obj
    if isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    if isinstance(obj, tuple):
        return tuple(replace_instances(*a, o) for o in obj)
    if isinstance(obj, list):
        return [replace_instances(*a, o) for o in obj]
    if isinstance(obj, dict):
        return {k: replace_instances(*a, v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj):
        return obj
    module = getattr(obj, "__module__", None)
    if module is None:
        module_cls = getattr(obj, "__class__", None)
        module = getattr(module_cls, "__module__", None)
    if module is not None:
        if module.startswith("jaxlib."):
            return obj
        if module in ("builtins", "torch", "numpy"):
            return obj
    if isinstance(obj, VirtualInstance):
        return registry[obj.id]
    registry[id(obj)] = obj
    return VirtualInstance.get_virtual_instance(obj)


def replace_instances_back(registry, backend, obj):
    if isinstance(obj, VirtualInstance):
        if obj.id in registry:
            return registry[obj.id]
        return obj.build_wrapper(backend)
    if obj is None:
        return obj
    if isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    if isinstance(obj, tuple):
        return tuple(replace_instances_back(registry, backend, o) for o in obj)
    if isinstance(obj, list):
        return [replace_instances_back(registry, backend, o) for o in obj]
    if isinstance(obj, dict):
        return {k: replace_instances_back(registry, backend, v) for k, v in obj.items()}
    return obj



class _RemoteCallable:
    def __init__(self, i):
        self.id = i


def replace_callables(obj, callables, depth=0):
    if callable(obj):
        is_host = getattr(obj, "__host__", depth <= 0)
        if is_host:
            callables.append(obj)
            return _RemoteCallable(len(callables) - 1)
        else:
            return obj
    if isinstance(obj, dict):
        return {k: replace_callables(v, callables, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return obj.__class__((replace_callables(v, callables, depth + 1) for v in obj))
    if is_dataclass(obj):
        return obj.__class__(**{k: replace_callables(v, callables, depth + 1) for k, v in obj.__dict__.items()})
    return obj


def inject_callables(obj: Any, send_message, my_id) -> Any:
    if isinstance(obj, _RemoteCallable):
        def callback(*args, **kwargs):
            send_message({"message": "callback", "thread_id": my_id, "callback": obj.id, "args": args, "kwargs": kwargs})

        return callback
    if isinstance(obj, dict):
        return {k: inject_callables(v, send_message, my_id) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)((inject_callables(v, send_message, my_id) for v in obj))
    if is_dataclass(obj):
        return type(obj)(**{k: inject_callables(v, send_message, my_id) for k, v in obj.__dict__.items()})
    return obj


class EventCancellationToken(CancellationToken):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cancelled_event = Event()

    def cancel(self):
        self._cancelled_event.set()
        super().cancel()

    @property
    def cancelled(self):
        return super().cancelled or self._cancelled_event.is_set()


class RPCWorker:
    def __init__(self):
        self._thread_executor = None

        self._instances = {}

    def __enter__(self):
        self._thread_executor = ThreadPoolExecutor(max_workers=8).__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._thread_executor is not None:
            self._thread_executor.__exit__(exc_type, exc_value, traceback)
            self._thread_executor = None
        self._instances = {}

    def process_message(self, msg, send_message):
        mid = msg["thread_id"]
        message = msg["message"]
        if message == "instance_del":
            logging.debug(f"Deleting instance {msg['instance']}")
            self._instances.pop(msg["instance"], None)
            return

        is_generator = False
        try:
            if message == "static_call":
                logging.debug(f"Calling function {msg['function']}")
                package, fnname = msg["function"].split(":", 1)
                fn = importlib.import_module(package)
                for x in fnname.split("."):
                    fn = getattr(fn, x)
                fn = getattr(fn, "__run_on_host_original__", fn)
                fn = cast(Callable, fn)
                kwargs = inject_callables(msg["kwargs"], send_message, mid)
                args = inject_callables(msg["args"], send_message, mid)
            elif message == "instance_call":
                logging.debug(f"Calling method {msg['name']} on instance {msg['instance']}")
                fn = self._instances[msg["instance"]]
                for x in msg["name"].split("."):
                    fn = getattr(fn, x)
                fn = cast(Callable, fn)
                kwargs = inject_callables(msg["kwargs"], send_message, mid)
                args = inject_callables(msg["args"], send_message, mid)
            elif message == "static_getattr":
                logging.debug(f"Obtaining property {msg['name']}")
                package, fnname = msg["name"].split(":", 1)
                obj = importlib.import_module(package)
                for x in fnname.split("."):
                    obj = getattr(obj, x)
                send_message({"message": "result", "thread_end": True, "thread_id": mid, "result": obj})
                return
            elif message == "instance_getattr":
                logging.debug(f"Obtaining property {msg['name']} on instance {msg['instance']}")
                obj = self._instances[msg["instance"]]
                for x in msg["name"].split("."):
                    obj = getattr(obj, x)
                send_message({"message": "result", "thread_end": True, "thread_id": mid, "result": obj})
                return
            else:
                raise RuntimeError(f"Unknown message {message}")

            if CancellationToken.current is not None:
                fn = cancellable(fn, cancellation_token=CancellationToken.current)
            result: Any = fn(*args, **kwargs)
            if inspect.isgeneratorfunction(fn):
                is_generator = True
                for r in result:
                    if CancellationToken.current is not None:
                        CancellationToken.current.raise_for_cancelled()
                    send_message({"message": "yield", "thread_id": mid, "yield": r, "is_generator": is_generator})
                result = None
            # We will register possible new instance and return the virtual instance
            result = replace_instances(self._instances, result)
            send_message({"message": "result", "thread_end": True, "thread_id": mid, "result": result, "is_generator": is_generator})
        except Exception as e:
            if not isinstance(e, CancelledException):
                traceback.print_exc()
            send_message({"message": "error", "thread_end": True, "thread_id": mid, "error": _remap_error(e), "is_generator": is_generator})

def generate_authkey():
    return secrets.token_hex(64).encode("ascii")


def run_worker(*, worker: Optional[RPCWorker] = None, address="localhost", port=None, authkey=None):
    if worker is None:
        with RPCWorker() as worker:
            return run_worker(worker=worker, address=address, port=port, authkey=authkey)
    else:
        assert port is not None, "Port must be provided"
        assert authkey is not None, "Authkey must be provided"

    with Client((address, port), authkey=authkey) as client, \
                        ThreadPoolExecutor(max_workers=8) as pool:
        recv = build_recv(client)
        send(client, {"message": "ready", "thread_end": True, "thread_id": -1})
        cancellation_tokens: Dict[int, CancellationToken] = {}
        out_queue = Queue()
        while not client.closed:
            try:
                outmsg = out_queue.get_nowait()
                if outmsg.get("thread_end", False):
                    cancellation_tokens.pop(outmsg["thread_id"], None)
                send(client, outmsg)
            except Empty:
                pass
            if not client.poll(0.0001):
                continue
            msg = recv()
            if msg["message"] == "close":
                send(client, {"message": "close_ack"})
                break
            mid = msg["thread_id"]
            if msg["message"] == "cancel":
                # Cancel without the current thread (perhaps a late message)
                if mid in cancellation_tokens:
                    cancellation_tokens.pop(mid).cancel()
                continue
            cancellation_token = cancellation_tokens.get(mid)
            if cancellation_token is None:
                cancellation_token = EventCancellationToken()
            cancellation_tokens[mid] = cancellation_token
            def process_message_with_token(token, *args, **kwargs):
                with (token or contextlib.nullcontext()):
                    worker.process_message(*args, **kwargs)
            pool.submit(
                process_message_with_token, cancellation_token, msg, lambda m: out_queue.put({**m, "thread_id": mid}))

                
def _listener_accept_with_cancel(listener: Listener, cancel_token: Optional[CancellationToken], timeout: float = 0):
    if cancel_token is None and timeout <= 0:
        return listener.accept()
    elif cancel_token is None:
        try:
            listener._listener._socket.settimeout(timeout)  # type: ignore
            return listener.accept()
        except socket.timeout:
            raise TimeoutError("Timeout waiting for connection")
    elif cancel_token is not None:
        start = time.time()
        wtimeout = 0.1
        if timeout > 0:
            wtimeout = min(wtimeout, timeout)
        while True:
            if cancel_token is not None:
                cancel_token.raise_for_cancelled()
            try:
                listener._listener._socket.settimeout(wtimeout)  # type: ignore
                return listener.accept()
            except socket.timeout:
                pass
            if timeout > 0 and time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for connection")
    assert False, "Unreachable"


class RPCMasterEndpoint:
    def __init__(self, *, address="localhost", port, authkey):
        self.address = address
        self.port = port
        self.authkey = authkey

        self._listener = None
        self._conn = None
        self._recv = None
        self._other_queues = {}

    @cancellable(mark_only=True)
    def send_message(self, message):
        # Start a thread
        cancellation_token = CancellationToken.current
        if self._conn is None or self._conn.closed:
            raise RuntimeError("There is no active connection.")
        assert self._recv is not None, "Not in a context"
        global _MESSAGE_COUNTER
        _MESSAGE_COUNTER += 1
        mid = _MESSAGE_COUNTER
        self._other_queues[mid] = Queue()

        send(self._conn, {**message, "thread_id": mid})

        has_ended = False
        cancel_send = False
        try:
            while True:
                if self._conn is None or self._conn.closed:
                    raise ConnectionError("Connection closed unexpectedly")
                if cancellation_token is not None and cancellation_token.cancelled:
                    if not cancel_send:
                        cancel_send = True
                        send(self._conn, {"message": "cancel", "thread_id": mid})
                msg = None
                if self._conn.poll(0.0001):
                    msg = self._recv()
                elif mid in self._other_queues:
                    try:
                        msg = self._other_queues[mid].get_nowait()
                    except Empty:
                        continue
                else:
                    continue

                _thread_id = msg.get("thread_id")
                if _thread_id != mid:
                    # If the message does not belong to this thread,
                    # create a queue for it and place it to the other thread queue
                    if mid in self._other_queues:
                        self._other_queues[mid].put(msg)
                    continue
                has_ended = has_ended or msg.get("thread_end", False)
                yield msg
                if has_ended:
                    break
        finally:
            # If not finished, end the thread
            self._other_queues.pop(mid, None)
            if not has_ended and not cancel_send:
                if self._conn is None or self._conn.closed:
                    raise ConnectionError("Connection closed unexpectedly")
                send(self._conn, {"message": "cancel", "thread_id": mid})

    @cancellable(mark_only=True)
    def __enter__(self):
        assert self._listener is None, "Already in a context"
        self._listener = Listener((self.address, self.port), authkey=self.authkey)
        return self

    @cancellable(mark_only=True)
    def wait_for_connection(self, timeout: float = 0):
        logging.info("Waiting for connection")
        assert self._listener is not None, "Not in a context"
        if self._conn is not None and not self._conn.closed:
            return

        start = time.time()
        conn = _listener_accept_with_cancel(self._listener, CancellationToken.current, timeout)
        logging.info(f"Connection accepted from {self._listener.last_accepted}")

        recv = build_recv(conn)
        if CancellationToken.current is not None:
            while True:
                if CancellationToken.current is not None:
                    CancellationToken.current.raise_for_cancelled()
                wait_for = 0.001
                if timeout > 0:
                    wait_for = min(0.001, start + timeout - time.time())
                    if wait_for <= 0:
                        raise TimeoutError("Timeout waiting for connection")
                if conn.poll(wait_for):
                    break
                sleep(wait_for)
        elif timeout > 0:
            wait_for = start + timeout - time.time()
            if wait_for <= 0 or not conn.poll(wait_for):
                raise TimeoutError("Timeout waiting for connection")
        msg = recv()
        assert msg["message"] == "ready", f"Unexpected message {msg['message']}"
        self._conn, self._recv = conn, recv

    def close(self):
        if self._conn is not None and self._recv is not None:
            if not self._conn.closed:
                send(self._conn, {"message": "close"})
                # Wait for close ack
                while self._conn is not None and not self._conn.closed:
                    try:
                        self._recv()
                    except (EOFError, BrokenPipeError):
                        break
            self._conn.close()
            self._conn = None
        self._recv = None
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class RPCBackend(Backend):
    def __init__(self, endpoint: RPCMasterEndpoint):
        self._endpoint = endpoint

    def _handle_thread(self, message, *args, **kwargs):

        # 1) Replace callables from the function call
        callables = []
        if message["message"] in {"static_call", "instance_call"}:
            args, kwargs = cast(Tuple[Any, Any], replace_callables((args, kwargs), callables, depth=-2))
            message["args"] = args
            message["kwargs"] = kwargs

        with EventCancellationToken(CancellationToken.current) as token:
            thread_iter = iter(self._endpoint.send_message(message))
            for message in thread_iter:
                if message.get("is_generator"):
                    original_message = message

                    def yield_fn():
                        try:
                            for message in chain([original_message], thread_iter):
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
                        except GeneratorExit:
                            # Cancel and wait for the thread to finish here
                            logging.debug("Generator closed before completion. Cancelling.")
                            token.cancel()

                            # Finish the thread
                            for message in thread_iter:
                                pass

                    return yield_fn()
                elif message["message"] == "error":
                    # Reset qpointer
                    raise message["error"]
                elif message["message"] == "callback":
                    callback = callables[message["callback"]]
                    callback(*message["args"], **message["kwargs"])
                    continue
                elif message["message"] == "result":
                    # Reset qpointer
                    out = message["result"]
                    # Replace instances
                    return replace_instances_back({}, self, out)
                else:
                    raise RuntimeError(f"Unknown message {message}")

    def static_getattr(self, attr: str) -> Any:
        return self._handle_thread({
            "message": "static_getattr", 
            "name": attr, 
        })

    def static_call(self, function: str, *args, **kwargs) -> Any:
        return self._handle_thread({
            "message": "static_call", 
            "function": function, 
            "is_cancellable": CancellationToken.current is not None,
        }, *args, **kwargs)

    def instance_call(self, instance: int, method: str, *args, **kwargs) -> Any:
        return self._handle_thread({
            "message": "instance_call", 
            "instance": instance, 
            "name": method, 
            "is_cancellable": CancellationToken.current is not None,
        }, *args, **kwargs)

    def instance_getattr(self, instance: int, name: str) -> Any:
        return self._handle_thread({
            "message": "instance_getattr", 
            "instance": instance, 
            "name": name})

    def instance_del(self, instance: int):
        try:
            return self._handle_thread({
                "message": "instance_del", 
                "instance": instance})
        except Exception as _:
            # The instance might have already been removed
            pass


_SAFE_ENV = (
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
    "NERFBASELINES_PREFIX",
)


def get_safe_environment():
    return os.environ.copy()
    # return {k: v for k, v in os.environ.items() if k.upper() in _SAFE_ENV}


class RemoteProcessRPCBackend(Backend):
    def __init__(self, address: str = "localhost", port: Optional[int] = None, python_path: Optional[str] = None):
        if port is None:
            port = random.randint(10000, 20000)

        self._address = address
        self._port = port
        self._python_path = python_path or sys.executable

        self._rpc_backend: Optional[RPCBackend] = None
        self._endpoint = None
        self._worker_running = False

        self._worker_process: Optional[subprocess.Popen] = None
        self._inside_context = False

    def __enter__(self):
        super().__enter__()
        self._inside_context = True
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self._inside_context = False
        self.close()

    def close(self):
        self._worker_running = False
        if self._endpoint is not None:
            self._endpoint.close()
            self._endpoint = None
        self._rpc_backend = None
        if self._worker_process is not None:
            if self._worker_process.poll() is None:
                try:
                    self._worker_process.wait(6)
                except subprocess.TimeoutExpired:
                    pass
            if self._worker_process.poll() is None:
                logging.info("Waiting for the server to shut down")
                try:
                    self._worker_process.wait(10)
                except subprocess.TimeoutExpired:
                    pass
            if self._worker_process.poll() is None:
                logging.error("Server did not shut down, killing")
                self._worker_process.kill()
                self._worker_process.wait()

    def _launch_worker(self, args, env):
        return subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL)

    def _ensure_started(self):
        if self._worker_running:
            return
        if not self._inside_context:
            raise RuntimeError("Cannot start the worker outside of a context")

        is_verbose = logging.getLogger().isEnabledFor(logging.DEBUG)
        nb = nerfbaselines.__name__
        code = f"""
import os
from {nb}.utils import setup_logging
from {run_worker.__module__} import {run_worker.__name__} as rw
setup_logging(verbose={is_verbose})
authkey = os.environ.pop("NB_AUTHKEY").encode("ascii")
rw(address="{self._address}", port={self._port}, authkey=authkey)
"""
        env = get_safe_environment()
        authkey = generate_authkey()
        package_path = Path(nerfbaselines.__file__).absolute().parent.parent
        env["PYTHONPATH"] = f'{package_path}:{env.get("PYTHONPATH", "")}'.rstrip(":")
        env["NB_AUTHKEY"] = authkey.decode("ascii")
        args = ["python", "-c", code]

        if self._endpoint is None:
            self._endpoint = RPCMasterEndpoint(address=self._address, port=self._port, authkey=authkey).__enter__()
        if self._rpc_backend is None:
            self._rpc_backend = RPCBackend(self._endpoint)
        self._worker_process = self._launch_worker(args, env)
        while True:
            try:
                if self._worker_process.poll() is not None:
                    raise RuntimeError(f"Worker died with status code {self._worker_process.poll()}")
                self._endpoint.wait_for_connection(1.)
                break
            except TimeoutError:
                continue
        self._worker_running = True
        logging.info("Backend worker started")

    def static_getattr(self, attr: str):
        self._ensure_started()
        assert self._rpc_backend is not None, "Backend not started"
        return self._rpc_backend.static_getattr(attr)

    def static_call(self, function: str, *args, **kwargs):
        self._ensure_started()
        assert self._rpc_backend is not None, "Backend not started"
        return self._rpc_backend.static_call(function, *args, **kwargs)

    def instance_call(self, instance: int, method: str, *args, **kwargs):
        self._ensure_started()
        assert self._rpc_backend is not None, "Backend not started"
        return self._rpc_backend.instance_call(instance, method, *args, **kwargs)
    
    def instance_getattr(self, instance: int, name: str):
        self._ensure_started()
        assert self._rpc_backend is not None, "Backend not started"
        return self._rpc_backend.instance_getattr(instance, name)
    
    def instance_del(self, instance: int):
        self._ensure_started()
        assert self._rpc_backend is not None, "Backend not started"
        return self._rpc_backend.instance_del(instance)