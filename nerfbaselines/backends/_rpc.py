import collections.abc
import threading
import io
import shutil
import tempfile
import functools
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
from threading import Event, Lock, Condition
from time import sleep
import pickle
import socket
from typing import Optional, List, Any, Dict, Callable, cast, Tuple, Type, Set
import inspect
import secrets
import logging
from multiprocessing import connection as mp_connection
from multiprocessing.connection import Listener, Client, Connection
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from ..utils import cancellable, CancellationToken, CancelledException
from ._common import Backend
import nerfbaselines


_MESSAGE_COUNTER = 0
MESSAGE_SIZE = 128 * 1024 * 1024  # 128 MB


def generate_authkey():
    return secrets.token_hex(64).encode("ascii")


def _remap_error(e: Exception):
    if e.__class__.__module__ == "builtins":
        return e
    elif e.__class__.__module__.startswith(_remap_error.__module__.split(".")[0]):
        return e

    # Remap exception
    return RuntimeError(f"Exception {e.__class__.__name__}: {e}")


def send(connection: Connection, message):
    msgo = message.copy()
    msgo.pop("kwargs", None)
    msgo.pop("args", None)
    # print("send", time.monotonic(), msgo)
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
        msgo = pickle.loads(message_bytes)
        msgo.pop("kwargs", None)
        msgo.pop("args", None)
        # print("recv", time.monotonic(), msgo)
        return pickle.loads(message_bytes)
    return recv


def customize_wrapper_separated_fs(local_shared_path, backend_shared_path, mounts, ns):
    def _translate_mounted_path(path):
        path = os.path.abspath(path)
        for local, remote in mounts:
            if path.startswith(local):
                return path.replace(local, remote)
        return None
    # Customize the wrapper for Method.
    # We check for simplified protocol
    if "save" in ns and "train_iteration" in ns and "render" in ns:
        # We replace the save with a custom save
        old_save = ns["save"].__func__
        @staticmethod
        @functools.wraps(old_save)
        def save(path: str):
            translated = _translate_mounted_path(path)
            if translated is not None:
                return old_save(path)

            with tempfile.TemporaryDirectory(prefix=local_shared_path + "/") as tmpdir:
                shutil.rmtree(tmpdir)
                shutil.copytree(path, tmpdir)
                remote_tmpdir = os.path.join(backend_shared_path, os.path.basename(tmpdir))
                out = old_save(remote_tmpdir)

                # Copy the files
                shutil.rmtree(path)
                shutil.copytree(tmpdir, path)
            return out
        ns["save"] = save

        # We replace the __call__ with a custom init
        old_init = ns["__call__"].__func__
        @staticmethod
        @functools.wraps(old_init)
        def __call__(*args, **kwargs):
            checkpoint = kwargs.get("checkpoint")
            if checkpoint is None:
                return old_init(*args, **kwargs)
            translated = _translate_mounted_path(checkpoint)
            if translated is not None:
                kwargs["checkpoint"] = translated
                return old_init(*args, **kwargs)
            with tempfile.TemporaryDirectory(prefix=local_shared_path + "/") as tmpdir:
                shutil.rmtree(tmpdir)
                shutil.copytree(checkpoint, tmpdir)
                remote_tmpdir = os.path.join(backend_shared_path, os.path.basename(tmpdir))
                kwargs["checkpoint"] = remote_tmpdir
                return old_init(*args, **kwargs)
        ns["__call__"] = __call__
    return ns


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

    def build_wrapper(self, backend: Backend, customize=None):
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
        ns["__class__"] = object
        if customize is not None:
            customize(ns)
        return types.new_class("VirtualInstanceRPC", (), {}, exec_body=lambda _ns: _ns.update(ns))()


def replace_instances(registry, used_instances, obj):
    a = registry, used_instances,
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
    if isinstance(obj, VirtualInstance):
        return registry[obj.id][0]
    if not isinstance(obj, collections.abc.Iterator):
        module = getattr(obj, "__module__", None)
        if module is None:
            module_cls = getattr(obj, "__class__", None)
            module = getattr(module_cls, "__module__", None)
        if module is not None:
            if module.startswith("jaxlib."):
                return obj
            if module in ("builtins", "torch", "numpy"):
                return obj
    if id(obj) not in registry:
        registry[id(obj)] = (obj, 1, set())
    else:
        obj, count, deps = registry[id(obj)]
        count += 1
        registry[id(obj)] = (obj, count, deps)
    used_instances.add(id(obj))
    return VirtualInstance.get_virtual_instance(obj)


def replace_instances_back(registry, backend, obj, customize):
    if isinstance(obj, VirtualInstance):
        if obj.id in registry:
            return registry[obj.id]
        return obj.build_wrapper(backend, customize=customize)
    if obj is None:
        return obj
    if isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    if isinstance(obj, tuple):
        return tuple(replace_instances_back(registry, backend, o, customize) for o in obj)
    if isinstance(obj, list):
        return [replace_instances_back(registry, backend, o, customize) for o in obj]
    if isinstance(obj, dict):
        return {k: replace_instances_back(registry, backend, v, customize) for k, v in obj.items()}
    return obj



class _RemoteCallable:
    def __init__(self, i):
        self.id = i


def replace_callables(obj, callables, depth=0):
    if callable(obj):
        is_host = getattr(obj, "__host__", depth <= 0)
        if is_host:
            callables[id(obj)] = obj
            return _RemoteCallable(id(obj))
        else:
            return obj
    if isinstance(obj, dict):
        return {k: replace_callables(v, callables, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return obj.__class__((replace_callables(v, callables, depth + 1) for v in obj))
    if is_dataclass(obj):
        return obj.__class__(**{k: replace_callables(v, callables, depth + 1) for k, v in obj.__dict__.items()})  # type: ignore
    return obj


def inject_callables(obj: Any, send_message, my_id=None) -> Any:
    if isinstance(obj, _RemoteCallable):
        def callback(*args, **kwargs):
            send_message({"message": "callback", "callback": obj.id, "args": args, "kwargs": kwargs})

        return callback
    if isinstance(obj, dict):
        return {k: inject_callables(v, send_message, my_id) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)((inject_callables(v, send_message, my_id) for v in obj))  # type: ignore
    if is_dataclass(obj):
        return type(obj)(**{k: inject_callables(v, send_message, my_id) for k, v in obj.__dict__.items()})  # type: ignore
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
        self._instances = {}
        self._client_instances = {}

    def _process_del(self, *, instance, **_):
        """
        ----del---->
        """
        logging.debug(f"Deleting instance {instance}")
        freed_instances = set()
        if instance in self._instances:
            instance_obj, count, deps = self._instances[instance]
            count -= 1
            if count <= 0:
                self._instances.pop(instance, None)
                for dep in deps:
                    if dep in self._client_instances:
                        self._client_instances[dep] -= 1
                        if self._client_instances[dep] <= 0:
                            self._client_instances.pop(dep)
                            freed_instances.add(dep)
            else:
                self._instances[instance] = (instance_obj, count, deps)
        return {
            "message": "del_ack",
            "thread_end": True,
            "freed_instances": list(freed_instances),
        }

    def _process_getattr(self, *, instance=None, name: str, **_):
        """
        ----getattr---->
        <---result------
        """
        try:
            if instance is None:
                logging.debug(f"Obtaining property {name}")
                package, fnname = name.split(":", 1)
                obj = importlib.import_module(package)
                for x in fnname.split("."):
                    obj = getattr(obj, x)
                return {"message": "result", "thread_end": True, "result": obj}
            else:
                logging.debug(f"Obtaining property {name} on instance {instance}")
                obj = self._instances[instance][0]
                for x in name.split("."):
                    obj = getattr(obj, x)
                return {"message": "result", "thread_end": True, "result": obj}
        except Exception as e:
            if not isinstance(e, CancelledException):
                traceback.print_exc()
            return {"message": "error", "thread_end": True, "error": _remap_error(e)}

    def _process_call(self, *, instance=None, name: str, kwargs, args, allocated_instances, cancellation_token_id=None, send_message, **_):
        """
        ----call---->
        <---result---
        """
        freed_instances = allocated_instances
        try:
            if instance is None:
                logging.debug(f"Calling function {name}")
                package, fnname = name.split(":", 1)
                fn = importlib.import_module(package)
                for x in fnname.split("."):
                    fn = getattr(fn, x)
                fn = getattr(fn, "__run_on_host_original__", fn)
                fn = cast(Callable, fn)
                kwargs = inject_callables(kwargs, send_message)
                args = inject_callables(args, send_message)
            else:
                logging.debug(f"Calling method {name} on instance {instance}")
                fn = self._instances[instance][0]
                for x in name.split("."):
                    fn = getattr(fn, x)
                fn = cast(Callable, fn)
                kwargs = inject_callables(kwargs, send_message)
                args = inject_callables(args, send_message)

            if cancellation_token_id is not None:
                cancellation_token = None
                if cancellation_token_id is not None:
                    cancellation_token, *_ = self._instances.get(cancellation_token_id, (None, None))
                    if cancellation_token is None:
                        cancellation_token = EventCancellationToken()
                        self._instances[cancellation_token_id] = (cancellation_token, 1, set())
                fn = cancellable(fn, cancellation_token=cancellation_token)
            result: Any = fn(*args, **kwargs)
            used_caller_instances = set()
            result = replace_instances(self._instances, used_caller_instances, result)
            if used_caller_instances:
                freed_instances = []
                # Add dependency between the returned object and the caller's instances
                for inst in used_caller_instances:
                    current_deps: Set[int] = self._instances[inst][2]
                    for dep in (set(allocated_instances) - current_deps):
                        current_deps.add(dep)
                        self._client_instances[dep] = self._client_instances.get(dep, 0) + 1
            return {"message": "result", "thread_end": True, "result": result, "freed_instances": freed_instances}
        except Exception as e:
            if not isinstance(e, (CancelledException, StopIteration)):
                traceback.print_exc()
            return {"message": "error", "thread_end": True, "error": _remap_error(e), "freed_instances": freed_instances}

    def cancel(self, cancellation_token_id):
        cancellation_token, *_ = self._instances.pop(cancellation_token_id, (None, None))
        if cancellation_token is not None:
            cancellation_token.cancel()

    def process_message(self, message, send_message):
        logging.debug(f"Processing message {message}")
        msg_type = message["message"]
        if msg_type == "del":
            outmsg = self._process_del(**message)
            if message.get("send_ack", True):
                send_message(outmsg)
        elif msg_type == "cancel":
            self.cancel(message["cancellation_token_id"])
        elif msg_type == "getattr":
            outmsg = self._process_getattr(**message)
            send_message(outmsg)
        elif msg_type == "call":
            outmsg = self._process_call(**message, send_message=send_message)
            send_message(outmsg)
        else:
            raise ValueError(f"Unknown message type {msg_type}")


def run_worker(*, worker: Optional[RPCWorker] = None, address="localhost", port=None, authkey=None):
    if worker is None:
        worker = RPCWorker()

    assert port is not None, "Port must be provided"
    assert authkey is not None, "Authkey must be provided"

    try:
        lock = threading.Lock()
        queue = Queue()

        def get_messages(queue, lock, conn: Connection, worker):
            try:
                _recv = build_recv(conn)
                while not conn.closed:
                    conn.poll(None)
                    with lock:
                        msg = _recv()
                    if msg["message"] == "close":
                        break
                    if msg["message"] == "cancel":
                        # Cancel without the current thread (perhaps a late message)
                        worker.cancel(msg["cancellation_token_id"])
                        continue
                    queue.put(msg)
            finally:
                queue.put({"message": "close"})
            
        with Listener((address, port), authkey=authkey) as listener:
            def recv():
                msg = queue.get()
                if msg["message"] == "close":
                    raise SystemExit
                return msg

            def send_callback(msg, callback=None):
                with lock:
                    send(conn, msg)
                if callback is not None:
                    submsg = recv()
                    while callback(submsg):
                        submsg = recv()

            with listener.accept() as conn:
                thread = threading.Thread(target=get_messages, daemon=True, args=(queue, lock, conn, worker))
                logging.info(f"Connection accepted from {listener.last_accepted}")
                send(conn, {"message": "ready", "thread_end": True})
                thread.start()
                while True:
                    worker.process_message(recv(), send_callback)
    except SystemExit:
        pass
    logging.info("Backend worker finished")


def _client_connect_with_cancel(address, authkey, cancel_token: Optional[CancellationToken], timeout: float = 0) -> Connection:
    def connect(timeout=None):
        family = mp_connection.address_type(address)  # type: ignore
        with socket.socket(getattr(socket, family)) as s:
            if timeout is not None:
                s.settimeout(timeout)
            s.setblocking(True)
            s.connect(address)
            if timeout is not None:
                s.settimeout(None)
            c = Connection(s.detach())
        mp_connection.answer_challenge(c, authkey)
        mp_connection.deliver_challenge(c, authkey)
        return c

    if cancel_token is None:
        start = time.time()
        while True:
            if timeout > 0 and time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for connection")
            try:
                return connect(timeout)
            except ConnectionRefusedError:
                continue
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
                return connect(wtimeout)
            except ConnectionRefusedError:
                pass
            except socket.timeout:
                pass
            if timeout > 0 and time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for connection")
    assert False, "Unreachable"



class _PipeCondition(threading.Condition):
    def __init__(self):
        super().__init__()
        self._condition = threading.Condition()
        self._read_fd, self._write_fd = os.pipe()

    def _consume(self):
        # while os.read(16):
        #     pass
        pass

    def fileno(self) -> int:
        return self._read_fd

    def notify(self, n=1):
        self._consume()
        os.write(self._write_fd, b"1")
        super().notify(n)

    def __del__(self):
        # Close the file descriptors
        if hasattr(self, "_read_fd"):
            os.close(self._read_fd)
            del self._read_fd
        if hasattr(self, "_write_fd"):
            os.close(self._write_fd)
            del self._write_fd


class RPCMasterEndpoint:
    def __init__(self, *, address="localhost", port, authkey):
        self.address = address
        self.port = port
        self.authkey = authkey

        self._conn = None
        self._recv = None
        self._other_threads_queue = Queue()
        self._other_threads_condition = _PipeCondition()
        self._main_thread = threading.get_ident()

    def __call__(self, message, callback=None):
        # Start a thread
        if self._conn is None or self._conn.closed:
            raise ConnectionError("There is no active connection.")
        assert self._recv is not None, "Not in a context"

        if threading.get_ident() != self._main_thread:
            assert callback is None, "callback must be None in threads other than main"
            with self._other_threads_condition:
                self._other_threads_queue.put(message)
                self._other_threads_condition.notify_all()
            return

        send(self._conn, {**message})
        if callback is not None:
            while True:
                if self._conn is None or self._conn.closed:
                    raise ConnectionError("Connection closed unexpectedly")
                readable = mp_connection.wait(
                    [self._conn] + [self._other_threads_condition.fileno()] if self._other_threads_condition is not None else []
                )
                try:
                    while True:
                        send(self._conn, self._other_threads_queue.get_nowait())
                except Empty:
                    pass
                if self._conn not in readable:
                    continue
                msg = self._recv()
                if not callback(msg):
                    break

    def __enter__(self):
        return self

    @cancellable(mark_only=True)
    def wait_for_connection(self, timeout: float = 0):
        if self._conn is not None and not self._conn.closed:
            return

        conn = _client_connect_with_cancel((self.address, self.port), self.authkey, CancellationToken.current, timeout)
        recv = build_recv(conn)
        msg = recv()
        assert msg["message"] == "ready", f"Unexpected message {msg['message']}"
        self._conn, self._recv = conn, recv

    def close(self):
        if self._conn is not None and self._recv is not None and not self._conn.closed:
            try:
                send(self._conn, {"message": "close"})
                # Wait for close ack
                while self._conn is not None and not self._conn.closed:
                    self._recv()
            except (EOFError, BrokenPipeError):
                pass
            self._conn.close()
        self._conn = None
        self._recv = None
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


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
    "CI",
    "CONDA_PKGS_DIRS",
    "PIP_CACHE_DIR",
    "TORCH_HOME",
    "NERFBASELINES_PREFIX",
)


def get_safe_environment():
    return os.environ.copy()
    # return {k: v for k, v in os.environ.items() if k.upper() in _SAFE_ENV}


class RPCBackend(Backend):
    def __init__(self, endpoint, customize_wrapper=None):
        self._customize_wrapper = customize_wrapper
        self._worker_instances = {}
        self._send = endpoint

    def _getattr(self, attr: str, instance):
        result_or_error = cast(Optional[Tuple[bool, Any]], None)
        def callback(msg):
            nonlocal result_or_error
            self._handle_free_instances(msg)
            if msg["message"] == "result":
                result_or_error = (False, msg["result"])
                return False
            elif msg["message"] == "error":
                result_or_error = (True, msg["error"])
                return False
            else:
                print("getattr", attr, instance, msg['message'])
                raise RuntimeError(f"Unexpected message {msg['message']}")
        self._send({
            "message": "getattr", 
            "instance": instance,
            "name": attr}, callback=callback)
        assert result_or_error is not None, "No result received"
        is_error, result = result_or_error
        if is_error:
            assert isinstance(result, BaseException)
            raise result
        else:
            return replace_instances_back({}, self, result, self._customize_wrapper)

    def static_getattr(self, attr: str) -> Any:
        return self._getattr(attr, None)

    def instance_getattr(self, instance: int, attr: str) -> Any:
        return self._getattr(attr, instance)

    def instance_del(self, instance: int):
        try:
            def callback(msg):
                self._handle_free_instances(msg)
                if msg["message"] != "del_ack":
                    raise RuntimeError(f"Unexpected message {msg['message']}")
            return self._send({
                "message": "del", 
                "instance": instance}, callback=callback)
        except ConnectionError:
            pass
        except Exception as _:
            traceback.print_exc()
            # The instance might have already been removed
            pass

    def _handle_free_instances(self, msg):
        freed_instances = msg.get("freed_instances", [])
        if freed_instances:
            for instance in freed_instances:
                self._worker_instances.pop(instance, None)

    def _call(self, function: str, instance, *args, **kwargs) -> Any:
        result_or_error = cast(Optional[Tuple[bool, Any]], None)
        def callback(msg):
            nonlocal result_or_error
            self._handle_free_instances(msg)
            if msg["message"] == "result":
                result_or_error = (False, msg["result"])
                return False
            elif msg["message"] == "error":
                result_or_error = (True, msg["error"])
                return False
            elif msg["message"] == "callback":
                callback = self._worker_instances[msg["callback"]]
                callback(*msg["args"], **msg["kwargs"])
                return True
            else:
                print(function, instance, msg['message'])
                raise RuntimeError(f"Unexpected message {msg['message']}")

        # 1) Replace callables from the function call
        callables = {}
        args, kwargs = cast(Tuple[Any, Any], replace_callables((args, kwargs), callables, depth=-2))
        self._worker_instances.update(callables)

        # 2) Add hook to the cancellation token
        cancellation_token_id = None
        cancellation_token = CancellationToken.current
        if cancellation_token is not None:
            cancellation_token_id = id(cancellation_token)
            cancellation_token.register_callback(
                lambda: self._send({
                    "message": "cancel", 
                    "cancellation_token_id": cancellation_token_id}))
            
            def del_hook(token):
                try:
                    # def callback(msg):
                    #     if msg["message"] != "del_ack":
                    #         raise RuntimeError(f"Unexpected message {msg['message']}")
                    return self._send({
                        "message": "del", 
                        "send_ack": False,
                        "instance": id(token)})#, callback=callback)
                except ConnectionError:
                    pass
                except Exception as _:
                    traceback.print_exc()
                    # The instance might have already been removed
                    pass
            cancellation_token.del_hooks.append(del_hook)
        message = {
            "message": "call", 
            "instance": instance, 
            "name": function, 
            "args": args,
            "kwargs": kwargs,
            "allocated_instances": list(callables.keys()),
            "cancellation_token_id": id(cancellation_token) if cancellation_token is not None else None,
        }
        self._send(message, callback=callback)
        assert result_or_error is not None, "No result received"
        is_error, result = result_or_error
        if is_error:
            assert isinstance(result, BaseException)
            raise result
        return replace_instances_back({}, self, result, self._customize_wrapper)

    def static_call(self, function: str, *args, **kwargs) -> Any:
        return self._call(function, None, *args, **kwargs)

    def instance_call(self, instance: int, method: str, *args, **kwargs) -> Any:
        return self._call(method, instance, *args, **kwargs)


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

    def _customize_wrapper(self, ns):
        return ns

    def _ensure_started(self):
        if self._worker_running:
            return
        if not self._inside_context:
            raise RuntimeError("Cannot start the worker outside of a context")

        is_verbose = logging.getLogger().isEnabledFor(logging.DEBUG)
        nb = nerfbaselines.__name__
        code = f"""
import os
# Hack for now to fix the cv2 failed import inside a thread.
# We should move to a fully sync model.
try:
    import cv2
except Exception:
    pass
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
            self._endpoint = RPCMasterEndpoint(
                address=self._address, 
                port=self._port, 
                authkey=authkey,
            ).__enter__()
        if self._rpc_backend is None:
            self._rpc_backend = RPCBackend(
                self._endpoint,
                customize_wrapper=self._customize_wrapper,
            )
        self._worker_process = self._launch_worker(args, env)
        logging.info("Waiting for connection")
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
    
    def instance_getattr(self, instance: int, attr: str):
        self._ensure_started()
        assert self._rpc_backend is not None, "Backend not started"
        return self._rpc_backend.instance_getattr(instance, attr)
    
    def instance_del(self, instance: int):
        self._ensure_started()
        assert self._rpc_backend is not None, "Backend not started"
        return self._rpc_backend.instance_del(instance)
