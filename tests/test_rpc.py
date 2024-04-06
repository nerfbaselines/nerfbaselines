import contextlib
from queue import Queue, Empty
import inspect
import threading
import random
import pytest
import time
from time import sleep
import gc
from typeguard import typeguard_ignore
from unittest import mock

from nerfbaselines.utils import CancelledException, CancellationToken
from nerfbaselines.backends._common import SimpleBackend
from nerfbaselines.backends._rpc import RPCWorker, RPCMasterEndpoint, EventCancellationToken, generate_authkey
from nerfbaselines.backends._rpc import run_worker, RPCBackend, RemoteProcessRPCBackend


def _test_function(a, b):
    return a + b


def cc(*args):
    def w():
        o = None
        for fn in args:
            o = fn()
        return o
    return w


class _TestObject:
    static_property = 8

    def __init__(self, a):
        self.a = a

    @staticmethod
    def static_method(a, b):
        return a + b

    @property
    def attr(self):
        return self.a + 1

    def test_method(self, b, *, c):
        self.a += 2
        return self.a - b + c

    @property
    def attr2(self):
        return self.a + 2

    def test_iter(self):
        for i in range(5):
            yield i

    def test_iter_callback(self, cb):
        for i in range(5):
            cb(i)
            yield i

    def test_iter_interrupt(self):
        for i in range(5):
            yield i
        raise RuntimeError("Not interrupted")


    def test_raise(self):
        raise ValueError("Test error")

    def test_cancel(self):
        for _ in range(1000):
            assert EventCancellationToken.current is not None
            sleep(1/1000)

    def test_cancel_iter(self):
        for i in range(1000):
            sleep(1/1000)
            yield i


def stream_callback_calls(fn):
    class _dummy:
        pass

    def wrapped(*args, **kwargs):
        q = Queue(maxsize=1)
        def job(token, *args, **kwargs):
            try:
                with token or contextlib.nullcontext():
                    try:
                        fn(lambda m: q.put(m), *args, **kwargs)
                    except Exception as e:
                        q.put(e)
            finally:
                q.put(_dummy)
        if args[0]["message"] == "instance_del":
            fn(lambda m: q.put(m), *args, **kwargs)
            return

        thread = threading.Thread(target=job, args=(EventCancellationToken.current,) + args, kwargs=kwargs, daemon=True)
        thread.start()

        # Consumer
        while True:
            try:
                next_item = q.get(timeout=0.0001)
                if next_item is _dummy:
                    break
                if isinstance(next_item, Exception):
                    raise next_item
                yield next_item
            except Empty:
                pass
        thread.join()
    return wrapped

GMID = 0


def stream_messages(fn):
    global GMID
    GMID += 1
    def run(callback, msg):
        def cb(msg):
            if isinstance(msg, dict) and msg.get("message") == "cancel_ack":
                callback(CancelledException())
            else:
                callback(msg)
        fn({**msg}, cb)
    return stream_callback_calls(run)


def _build_worker_send(worker):
    m_callback = None
    def send(msg, callback=None):
        nonlocal m_callback
        if callback is not None:
            m_callback = callback
        is_active = True
        def send_back(msg_back, callback=None):
            nonlocal is_active
            assert m_callback is not None
            if m_callback(msg_back):
                is_active = False
            assert callback is None
        worker.process_message(msg, send_back if callback is not None else None)
    return send


@typeguard_ignore
def test_rpc_backend_static_function():
    worker = RPCWorker()
    endpoint = mock.MagicMock(side_effect=_build_worker_send(worker))
    backend = RPCBackend(endpoint=endpoint)

    # Test simple call
    func = backend.wrap(_test_function)
    out = func(1, 2)
    endpoint.assert_called_once()
    assert out == 3
    endpoint.reset_mock()

    # Test simple call
    out = func(1, 3)
    endpoint.assert_called_once()
    assert out == 4
    endpoint.reset_mock()


@typeguard_ignore
def test_rpc_backend_instance():
    worker = RPCWorker()
    endpoint = mock.MagicMock(side_effect=_build_worker_send(worker))
    backend = RPCBackend(endpoint=endpoint)

    testobj = backend.wrap(_TestObject)
    assert inspect.isclass(testobj)
    out = testobj(1)
    assert len(worker._instances) == 1

    # Test simple call
    endpoint.assert_called_once()
    assert out.test_method(5, c=3) == 1
    endpoint.reset_mock()
    assert out.test_method(5, c=3) == 3
    endpoint.reset_mock()

    # Test attr accessl
    assert out.attr == 6
    endpoint.assert_called_once()
    endpoint.reset_mock()

    # Test different attr
    assert out.attr2 == 7
    endpoint.assert_called_once()
    endpoint.reset_mock()

    # Test raise error
    with pytest.raises(ValueError) as e:
        out.test_raise()
        e.match("Test error")

    # Test instance was deleted
    del out
    gc.collect()
    assert not any(worker._instances)


def test_rpc_backend_yield():
    worker = RPCWorker()
    endpoint = mock.MagicMock(side_effect=_build_worker_send(worker))
    backend = RPCBackend(endpoint=endpoint)
    inst = backend.wrap(_TestObject)(1)

    # Test simple call
    assert list(inst.test_iter()) == list(range(5))

    # Test interrupt iterator
    with pytest.raises(RuntimeError):
        list(inst.test_iter_interrupt())
    for i in inst.test_iter_interrupt():
        if i > 2:
            break

    class _iterator():
        def __iter__(self):
            self.i = -1
            return self

        def __next__(self):
            if self.i > 3:
                raise StopIteration
            self.i += 1
            return self.i
    with mock.patch.object(_TestObject, "test_iter", lambda self: _iterator()):
        # Test simple call
        out = []
        for i in inst.test_iter():
            assert len(worker._instances) == 2
            out.append(i)
        assert out == list(range(5))
        assert len(worker._instances) == 1

    # Test if passes isgenerator test
    import collections.abc
    iterator = inst.test_iter()
    assert isinstance(iterator, collections.abc.Iterator)

    del inst, iterator
    gc.collect()
    assert len(worker._instances) == 0


@typeguard_ignore
def test_rpc_backend_cancel():
    worker = RPCWorker()
    endpoint = mock.MagicMock(side_effect=_build_worker_send(worker))
    backend = RPCBackend(endpoint=endpoint)
    assert len(worker._instances) == 0
    inst = backend.wrap(_TestObject)(1)
    assert len(worker._instances) == 1

    # Test cancel
    start = time.time()
    with CancellationToken() as token:
        inst.static_method(1, 2)
        assert len(worker._instances) == 2
    del token
    gc.collect()
    assert len(worker._instances) == 1

    # Test cancel
    start = time.time()
    with CancellationToken() as token:
        thread = threading.Thread(target=lambda: sleep(0.001) or token.cancel(), daemon=True)
        thread.start()
        with pytest.raises(CancelledException):
            inst.test_cancel()
        thread.join()
        assert time.time() - start < 0.1
    assert len(worker._instances) == 1

    # Test cancel iterable
    start = time.perf_counter()
    with CancellationToken() as token:
        thread = threading.Thread(target=lambda: sleep(0.001) or token.cancel(), daemon=True)
        thread.start()
        with pytest.raises(CancelledException):
            list(inst.test_cancel_iter())
        thread.join()
    assert time.perf_counter() - start < 0.1

    del inst
    gc.collect()
    assert len(worker._instances) == 0

def test_master_endpoint_cancel_waiting_for_connection():
    port = random.randint(10000, 20000)
    authkey = generate_authkey()
    start = time.time()
    with RPCMasterEndpoint(port=port, authkey=authkey) as endpoint:
        # Test timeout
        with pytest.raises(TimeoutError):
            endpoint.wait_for_connection(timeout=0.01)
        assert time.time() - start < 0.1

        # Test connection ready after first timeout
        start = time.time()
        with pytest.raises(TimeoutError):
            endpoint.wait_for_connection(timeout=0.01)
        assert time.time() - start < 0.1

        # Test cancel
        start = time.time()
        with pytest.raises(CancelledException):
            with EventCancellationToken() as token:
                threading.Thread(target=lambda: token.cancel(*[sleep(0.01)][:0]), daemon=True).start()
                endpoint.wait_for_connection()
        assert time.time() - start < 0.5

        # Test cancel and timeout
        start = time.time()
        with pytest.raises(TimeoutError):
            with EventCancellationToken() as token:
                threading.Thread(target=lambda: token.cancel(*[sleep(0.015)][:0]), daemon=True).start()
                endpoint.wait_for_connection(0.01)
        assert time.time() - start < 0.1


def test_master_endpoint_run_worker():
    worker = mock.MagicMock(spec=RPCWorker)
    received_messages = []
    def process_message(msg, callback):
        received_messages.append(msg)
        callback({"message": "res1"})
        callback({"message": "res2", "thread_end": True})

    worker.process_message = mock.MagicMock(side_effect=process_message)
    port = random.randint(10000, 20000)
    authkey = generate_authkey()
    exc = []

    def run_worker_():
        try:
            run_worker(worker=worker, port=port, authkey=authkey)
        except Exception as e:
            import traceback
            traceback.print_exc()
            exc.append(e)

    with RPCMasterEndpoint(port=port, authkey=authkey) as endpoint:
        thread = threading.Thread(target=run_worker_, daemon=True)
        thread.start()

        sleep(0.05)
        assert thread.is_alive()

        endpoint.wait_for_connection()
        result = []
        def callback(r):
            result.append(r)
            return not r.get("thread_end", False)
        endpoint({"message": "test"}, callback)
        assert len(result) == 2
        assert len(received_messages) == 1
        assert result[0]["message"] == "res1"
        assert result[1]["message"] == "res2"

    if exc:
        raise exc[0]
    worker.process_message.assert_called_once()

    sleep(0.05)
    assert not thread.is_alive()
    assert len(received_messages) == 1
    
    thread.join()

    # Test cancel
    received_messages, result = [], []
    was_cancelled = False
    def process_message_cancel(msg, callback):
        callback({**msg, "message": "back-" + msg["message"]})
        if msg["message"] == "test":
            callback({"message": "res1"})
            for _ in range(5000):
                if was_cancelled:
                    break
                sleep(0.01)
            callback({"message": "res2", "thread_end": True})
    worker.process_message = mock.MagicMock(side_effect=process_message_cancel)
    def cancel(x):
        nonlocal was_cancelled
        was_cancelled = True
        result.append({"message": "cancel", "id": x})
    worker.cancel = mock.MagicMock(side_effect=cancel)
    with RPCMasterEndpoint(port=port, authkey=authkey) as endpoint:
        thread = threading.Thread(target=run_worker_, daemon=True)
        thread.start()
        endpoint.wait_for_connection()
        threading.Thread(target=lambda: time.sleep(0.05) or endpoint({
            "message": "cancel",
            "cancellation_token_id": 12,
        }), daemon=True).start()
        def callback(r):
            result.append(r)
            return not r.get("thread_end", False)
        endpoint({"message": "test"}, callback)
        assert was_cancelled
        worker.cancel.assert_called()
        print(result)
        assert len(result) == 4
        assert result[0]["message"] == "back-test"
        assert result[1]["message"] == "res1"
        assert result[2]["message"] == "cancel"
        assert result[2]["id"] == 12
        assert result[3]["message"] == "res2"
        if exc:
            raise exc[0]
    worker.process_message.assert_called_once()

    sleep(0.05)
    assert not thread.is_alive()
    
    thread.join()


@typeguard_ignore
def test_master_endpoint_broken_connection():
    worker = mock.MagicMock(spec=RPCWorker)
    received_messages = []
    def process_message(msg, send):
        received_messages.append(msg)
        for _ in range(100):
            sleep(0.01)
        send({"message": "res1", "thread_end": True})

    worker.process_message = mock.MagicMock(side_effect=process_message)
    port = random.randint(10000, 20000)
    authkey = generate_authkey()
    exc = []
    def run_worker_():
        try:
            run_worker(worker=worker, port=port, authkey=authkey)
        except Exception as e:
            exc.append(e)

    # Test cancel
    received_messages, result = [], []
    with RPCMasterEndpoint(port=port, authkey=authkey) as endpoint:
        thread = threading.Thread(target=run_worker_, daemon=True)
        thread.start()
        endpoint.wait_for_connection()
        def break_connection():
            assert endpoint._conn is not None
            endpoint._conn.close()
        threading.Thread(target=lambda: time.sleep(0.1) or break_connection(), daemon=True).start()
        with pytest.raises(OSError):
            def callback(r):
                result.append(r)
                return not r.get("thread_end", False)
            endpoint({"message": "test"}, callback)
        if exc:
            raise exc[0]
        assert len(result) == 0
        assert len(received_messages) == 1
    worker.process_message.assert_called_once()

    sleep(0.05)
    assert not thread.is_alive()
    assert len(received_messages) == 1


def test_simple_backend_static_function():
    backend = SimpleBackend()

    # Test simple call
    assert backend.wrap(_test_function)(1, 2) == 3


def test_simple_backend_instance():
    backend = SimpleBackend()

    out = backend.wrap(_TestObject)(1)

    # Test simple call
    assert out.test_method(5, c=3) == 1
    assert out.test_method(5, c=3) == 3

    # Test attr accessl
    assert out.attr == 6


@typeguard_ignore
def test_remote_process_rpc_backend():
    with RemoteProcessRPCBackend() as backend:
        # Test simple call
        assert backend.wrap(_test_function)(1, 2) == 3

        out = backend.wrap(_TestObject)(1)

        # Test simple call
        assert out.test_method(5, c=3) == 1
        assert out.test_method(5, c=3) == 3

        # Test attr accessl
        assert out.attr == 6


def _test_iterable_stop_iteration():
    for i in range(10):
        yield i
        sleep(0.001)
    _test_iterable_stop_iteration.val = True  # type: ignore
_test_iterable_stop_iteration.val = False  # type: ignore


@typeguard_ignore
def test_rpc_backend_iterator_closed_propagate():
    port = random.randint(10000, 20000)
    authkey = generate_authkey()

    def rt(port, authkey):
        try:
            run_worker(port=port, authkey=authkey)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
    worker_thread = threading.Thread(target=rt, kwargs={"port": port, "authkey": authkey}, daemon=True)
    worker_thread.start()

    with RPCMasterEndpoint(port=port, authkey=authkey) as endpoint, RPCBackend(endpoint=endpoint) as backend:
        endpoint.wait_for_connection(timeout=1.0)

        _test_iterable_stop_iteration.val = False  # type: ignore
        next(iter(backend.static_call(_test_iterable_stop_iteration.__module__+":"+_test_iterable_stop_iteration.__name__)))
        assert not _test_iterable_stop_iteration.val  # type: ignore

        _test_iterable_stop_iteration.val = False  # type: ignore
        for _ in backend.static_call(_test_iterable_stop_iteration.__module__+":"+_test_iterable_stop_iteration.__name__):
            pass
        assert _test_iterable_stop_iteration.val  # type: ignore

    worker_thread.join()


def test_rpc_backend_yield_callback():
    worker = RPCWorker()
    endpoint = mock.MagicMock(side_effect=_build_worker_send(worker))
    backend = RPCBackend(endpoint=endpoint)
    inst = backend.wrap(_TestObject)(1)

    callback = mock.MagicMock()

    i = -1
    assert len(backend._worker_instances) == 0
    for i in inst.test_iter_callback(callback):
        assert len(backend._worker_instances) == 1
        callback.assert_called_with(i)
    assert i == 4
    assert len(backend._worker_instances) == 0


    # Test if the allocated callback is properly removed
    assert len(backend._worker_instances) == 0
    iterator = inst.test_iter_callback(callback)
    assert len(backend._worker_instances) == 1
    del iterator
    gc.collect()
    assert len(backend._worker_instances) == 0

    del inst
    gc.collect()
    assert len(worker._instances) == 0


def test_rpc_backend_remote_yield_callback():
    port = random.randint(10000, 20000)
    authkey = generate_authkey()

    def rt(port, authkey):
        try:
            run_worker(port=port, authkey=authkey)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
    worker_thread = threading.Thread(target=rt, kwargs={"port": port, "authkey": authkey}, daemon=True)
    worker_thread.start()

    with RPCMasterEndpoint(port=port, authkey=authkey) as endpoint, RPCBackend(endpoint=endpoint) as backend:
        endpoint.wait_for_connection(timeout=1.0)

        inst = backend.wrap(_TestObject)(1)

        callback = mock.MagicMock()

        i = -1
        for i in inst.test_iter_callback(callback):
            callback.assert_called_with(i)
        assert i == 4

    worker_thread.join()