import pprint
from contextlib import nullcontext
import collections.abc
import threading
import shutil
import tempfile
import functools
import traceback
import sys
import subprocess
import os
import dataclasses
from functools import partial
import types
from dataclasses import dataclass
from typing import Optional, List, Any
import inspect
import logging
from queue import Queue
from ..utils import CancellationToken, CancelledException
from . import _common


def _remap_error(e: BaseException):
    if e.__class__.__module__ == "builtins":
        return e
    elif e.__class__.__module__.startswith(_remap_error.__module__.split(".")[0]):
        return e

    # Remap exception
    if isinstance(e, Exception):
        return RuntimeError(f"Exception {e.__class__.__name__}: {e}")
    return BaseException(f"BaseException {e.__class__.__name__}: {e}")


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
class _VirtualInstance:
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
        return _VirtualInstance(id(obj), 
                               methods=methods, 
                               attrs=attrs)

    def build_wrapper(self, backend: _common.Backend, customize=None):
        instance_id = self.id
        ns = {}
        for k in self.methods:
            ns[k] = staticmethod(partial(backend.instance_call, instance_id, k))
        ns["__call__"] = staticmethod(partial(backend.instance_call, instance_id, "__call__"))
        ns["__repr__"] = staticmethod(lambda: f"<VirtualInstance {instance_id}>")
        ns["__str__"] = staticmethod(lambda: f"<VirtualInstance {instance_id}>")
        ns["__del__"] = staticmethod(partial(backend.instance_del, instance_id))
        ns["__class__"] = object
        ns["__nb_virtual_instance__"] = self
        if customize is not None:
            customize(ns)
        return types.new_class("_VirtualInstanceRPC", (), {}, exec_body=lambda _ns: _ns.update(ns))()


def _replace_instances(registry, obj):
    a = registry,
    if obj is None:
        return obj
    if isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    if isinstance(obj, tuple):
        return tuple(_replace_instances(*a, o) for o in obj)
    if isinstance(obj, list):
        return [_replace_instances(*a, o) for o in obj]
    if isinstance(obj, dict):
        return {k: _replace_instances(*a, v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj):
        return obj
    if isinstance(obj, _VirtualInstance):
        return registry[obj.id]
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
        registry[id(obj)] = obj
    return _VirtualInstance.get_virtual_instance(obj)


def _is_generator(out):
    if isinstance(out, types.GeneratorType):
        return True
    if isinstance(out, collections.abc.Iterator):
        return True
    return False


class RPCWorker:
    def __init__(self):
        self._cancellation_tokens = {}
        self._backend = _common.SimpleBackend()
        self._instances = self._backend._instances

    def _process_del(self, *, instance, **_):
        """
        ----del---->
        """
        logging.debug(f"Deleting instance {instance}")
        self._backend.instance_del(instance)
        return { "message": "del_ack" }

    def _process_call(self, *, instance=None, name: str, kwargs, args, cancellation_token_id=None, **_):
        try:
            if instance is None:
                fn = partial(self._backend.static_call, name)
            else:
                fn = partial(self._backend.instance_call, instance, name)

            # Resolve virtual instances
            args = tuple(self._instances[x.id] if isinstance(x, _VirtualInstance) else x for x in args)
            kwargs = {k:self._instances[x.id] if isinstance(x, _VirtualInstance) else x for k, x in kwargs.items()}

            if cancellation_token_id is not None:
                cancel_context = CancellationToken()
                self._cancellation_tokens[cancellation_token_id] = cancel_context
            else:
                cancel_context = nullcontext()
            try:
                with cancel_context:
                    result: Any = fn(*args, **kwargs)
            except StopIteration as e:
                if instance is not None:
                    self._instances.pop(instance, None)
                raise e
            out = {}

            # Handle iterators (to make them faster)
            if _is_generator(result):
                out = {}
                out["message"] = "iterable_result"
                out["errors"] = {}
                try:
                    with cancel_context:
                        iterator = iter(result)
                    out["iterator"] = id(iterator)
                    self._instances[id(iterator)] = iterator
                    try:
                        with cancel_context:
                            next_result = next(iterator)
                        out["next_result"] = next_result,
                    except StopIteration as e:
                        self._instances.pop(id(iterator), None)
                        out["errors"]["__next__"] = _remap_error(e)
                        return out
                    except BaseException as e:
                        out["errors"]["__next__"] = _remap_error(e)
                        return out
                except BaseException as e:
                    out["errors"]["__iter__"] = _remap_error(e)
                    return out
                return out
            else:
                out = {"message": "result", 
                       "result": _replace_instances(self._instances, result)}

            return out
        except BaseException as e:
            if not isinstance(e, (CancelledException, StopIteration, KeyboardInterrupt)):
                traceback.print_exc()
            return {"message": "error", "error": _remap_error(e)}
        finally:
            if cancellation_token_id is not None:
                self._cancellation_tokens.pop(cancellation_token_id, None)

    def cancel(self, cancellation_token_id):
        cancellation_token = self._cancellation_tokens.get(cancellation_token_id, None)
        if cancellation_token is not None:
            cancellation_token.cancel()

    def handle_interrupt(self, message):
        logging.debug(f"Processing interrupt message {message}")
        msg_type = message["message"]
        if msg_type == "cancel":
            self.cancel(message["cancellation_token_id"])
        else:
            raise RuntimeError(f"Unknown message type {msg_type}")

    def handle(self, message):
        logging.debug(f"Processing message {message}")
        msg_type = message["message"]
        if msg_type == "del":
            return self._process_del(**message)
        elif msg_type == "call":
            return self._process_call(**message)
        else:
            return {
                "message": "error",
                "error": RuntimeError(f"Unknown message type {msg_type}")
            }


def run_worker(*, protocol):
    interrupt_result_queue = Queue()
    rpc_worker = RPCWorker()
    handle = rpc_worker.handle
    handle_interrupt = rpc_worker.handle_interrupt

    def worker_interrupt(protocol, handle_interrupt, interrupt_result_queue):
        try:
            while True:
                msg = protocol.receive(channel=1)
                handle_interrupt(msg)
        except BaseException as e:
            interrupt_result_queue.put(e)
            return

    safe_terminate = False
    try:
        protocol.connect_worker()
        interrupt_thread = threading.Thread(
            target=worker_interrupt, 
            args=(protocol, handle_interrupt, interrupt_result_queue))
        logging.info(f"Connection accepted, protocol: {protocol.protocol_name}")
        interrupt_thread.start()
        while interrupt_thread.is_alive():
            msg = protocol.receive(channel=0)
            if msg.get("message") == "_safe_close":
                safe_terminate = True
                protocol.close()
                break
            try:
                outmsg = handle(msg)
            except BaseException as e:
                outmsg = {"message": "error", "error": RuntimeError("Unhandled error: "+str(e))}
            protocol.send(outmsg)
        interrupt_thread.join()
        if not safe_terminate:
            raise interrupt_result_queue.get()
    except ConnectionError as e:
        logging.debug("Connection closed with error", exc_info=e)
        logging.warning("Backend worker disconnected")
    except KeyboardInterrupt:
        pass
    finally:
        protocol.close()
    logging.info("Backend worker finished")


def get_safe_environment():
    return os.environ.copy()


class _IterableResultProxy:
    def __init__(self, backend, iterator_id, next_result, errors):
        self._iterator_id = iterator_id
        self._next_result = next_result
        self._errors = errors
        self._backend = backend

    def __iter__(self):
        if "__iter__" in self._errors:
            raise self._errors["__iter__"]
        return self

    def __next__(self):
        try:
            if "__next__" in self._errors:
                raise self._errors["__next__"]
            if self._next_result is not None:
                out = self._next_result[0]
                self._next_result = None
                return out
            if self._iterator_id is None:
                raise RuntimeError("Iterator is closed")
            CancellationToken.cancel_if_requested()
            return self._backend.instance_call(self._iterator_id, "__next__")
        except StopIteration:
            self._iterator_id = None
            raise

    def close(self):
        if self._iterator_id is not None:
            self._backend.instance_del(self._iterator_id)
            self._iterator_id = None

    def __del__(self):
        self.close()


class RPCBackend(_common.Backend):
    def __init__(self, protocol, customize_wrapper=None):
        self._protocol = protocol
        self._customize_wrapper = customize_wrapper
        self._remote_instances_counter = {}
        self._interrupt_lock = threading.Lock()
        self._main_lock = threading.Lock()

    def _send_interrupt(self, message):
        with self._interrupt_lock:
            self._protocol.send(message, channel=1)

    def _send(self, message, zero_copy=False):
        with self._main_lock:
            self._protocol.send(message)
            return self._protocol.receive(channel=0, zero_copy=zero_copy)

    def instance_del(self, instance: int):
        try:
            count = self._remote_instances_counter.get(instance, 0)
            count = max(0, count-1)
            if count == 0:
                msg = self._send({
                    "message": "del", 
                    "instance": instance})
                self._remote_instances_counter.pop(instance, None)
                if msg["message"] != "del_ack":
                    raise RuntimeError(f"Unexpected message {msg['message']}")
        except Exception as _:
            # The instance might have already been removed
            pass

    def _fix_backend_for_virtual_instance(self, obj):
        return getattr(type(obj), "__nb_virtual_instance__", obj)

    def _call(self, function: str, instance, *args, **kwargs) -> Any:
        with _common.set_allocator(self._protocol.get_allocator(0)):
            # 2) Add hook to the cancellation token
            cancellation_token_id = None
            cancellation_token = CancellationToken.current
            cancel_callback = None
            try:
                if cancellation_token is not None:
                    cancellation_token_id = id(cancellation_token)
                    cancel_callback = lambda: self._send_interrupt({
                        "message": "cancel", 
                        "cancellation_token_id": cancellation_token_id})
                    cancellation_token._callbacks.append(cancel_callback)

                args = tuple(self._fix_backend_for_virtual_instance(x) for x in args)
                kwargs = {k:self._fix_backend_for_virtual_instance(x) for k, x in kwargs.items()}

                message = {
                    "message": "call", 
                    "instance": instance, 
                    "name": function, 
                    "args": args,
                    "kwargs": kwargs,
                    "cancellation_token_id": id(cancellation_token) if cancellation_token is not None else None,
                }
                zero_copy = _common.current_backend_options().zero_copy
                msg = self._send(message, zero_copy=zero_copy)
            finally:
                if cancellation_token is not None and cancel_callback is not None:
                    cancellation_token._callbacks.remove(cancel_callback)
            if msg["message"] == "iterable_result":
                if msg.get("iterator_id") is not None:
                    self._remote_instances_counter[msg["iterator_id"]] = 1
                return _IterableResultProxy(
                    self,
                    msg.get("iterator"),
                    msg.get("next_result"),
                    msg.get("errors", {}))
            elif msg["message"] == "result":
                result = msg["result"]
                if isinstance(result, _VirtualInstance):
                    self._remote_instances_counter[result.id] = self._remote_instances_counter.get(result.id, 0) + 1
                    return result.build_wrapper(self, customize=self._customize_wrapper)
                return result
            elif msg["message"] == "error":
                raise msg["error"]
            else:
                print(function, instance, msg['message'])
                raise RuntimeError(f"Unexpected message {msg['message']}")

    def static_call(self, function: str, *args, **kwargs) -> Any:
        return self._call(function, None, *args, **kwargs)

    def instance_call(self, instance: int, method: str, *args, **kwargs) -> Any:
        return self._call(method, instance, *args, **kwargs)


class RemoteProcessRPCBackend(_common.Backend):
    def __init__(self, *, python_path: Optional[str] = None, protocol=None):
        self._python_path = python_path or sys.executable

        self._rpc_backend: Optional[RPCBackend] = None
        self._protocol = protocol
        self._worker_running = False

        self._worker_process: Optional[subprocess.Popen] = None
        self._worker_monitor_thread = None
        self._inside_context = False

    def __enter__(self):
        super().__enter__()
        self._inside_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If there was no exception, we safely close the worker by sending a close message
        force = False
        if isinstance(exc_val, (ConnectionError, BrokenPipeError, OSError, TimeoutError)):
            force = True
        self._inside_context = False
        self.close(_force=force)
        super().__exit__(exc_type, exc_val, exc_tb)

    def close(self, _force=False):
        if not _force:
            try:
                if self._protocol is not None and self._worker_running:
                    self._protocol.send({"message": "_safe_close"})
            except Exception:
                pass

        self._worker_running = False
        if self._protocol is not None:
            self._protocol.close()
            self._protocol = None
        if self._worker_monitor_thread is not None:
            self._worker_monitor_thread.join()
            self._worker_monitor_thread = None
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

    def _worker_monitor(self):
        try:
            while self._worker_running and self._worker_process is not None:
                status = self._worker_process.poll()
                if status is not None and self._worker_running and self._worker_process is not None:
                    logging.error(f"Worker died with status code {self._worker_process.poll()}")

                    # Now, we attenpt to kill the worker by closing the protocol
                    if self._protocol is not None:
                        self._protocol.close()
                if status is not None:
                    break
        except BaseException:
            pass

    def _ensure_started(self):
        if self._worker_running:
            return
        if not self._inside_context:
            raise RuntimeError("Cannot start the worker outside of a context")

        is_verbose = logging.getLogger().isEnabledFor(logging.DEBUG)
        if self._protocol is None:
            from ._transport_protocol import TransportProtocol
            self._protocol = TransportProtocol()

        self._protocol.start_host()
        protocol_kwargs = self._protocol.get_worker_configuration()
        init_protocol_code = ", ".join(
            f"{k}={pprint.pformat(v)}"
            for k, v in protocol_kwargs.items()
        )
        code = f"""
import os
# Hack for now to fix the cv2 failed import inside a thread.
# We should move to a fully sync model.
try:
    import cv2
except Exception:
    pass
from nerfbaselines.backends._common import setup_logging
setup_logging(verbose={is_verbose})
from {run_worker.__module__} import {run_worker.__name__} as rw
from {self._protocol.__class__.__module__} import {self._protocol.__class__.__name__} as P
rw(protocol=P({init_protocol_code}))
"""
        env = get_safe_environment()
        args = ["python", "-c", code]

        self._worker_process = self._launch_worker(args, env)
        logging.info("Waiting for connection")
        while True:
            try:
                if self._worker_process.poll() is not None:
                    raise RuntimeError(f"Worker died with status code {self._worker_process.poll()}")
                self._protocol.wait_for_worker(timeout=8.)
                break
            except TimeoutError:
                continue
        self._worker_running = True

        # Start monitor thread
        self._worker_monitor_thread = threading.Thread(target=self._worker_monitor)
        self._worker_monitor_thread.start()
        
        logging.info("Backend worker started")

        if self._rpc_backend is None:
            self._rpc_backend = RPCBackend(
                self._protocol,
                customize_wrapper=self._customize_wrapper,
            )

    def static_call(self, function: str, *args, **kwargs):
        self._ensure_started()
        assert self._rpc_backend is not None, "Backend not started"
        return self._rpc_backend.static_call(function, *args, **kwargs)

    def instance_call(self, instance: int, method: str, *args, **kwargs):
        self._ensure_started()
        assert self._rpc_backend is not None, "Backend not started"
        return self._rpc_backend.instance_call(instance, method, *args, **kwargs)
    
    def instance_del(self, instance: int):
        self._ensure_started()
        assert self._rpc_backend is not None, "Backend not started"
        return self._rpc_backend.instance_del(instance)
