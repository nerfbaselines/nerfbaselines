import contextlib
import threading
import pytest
import functools
import signal
from nerfbaselines.backends._rpc import _transport_protocols_registry


def _signal_handler(signum, frame):
    del signum, frame
    pytest.fail("Timeout")


def timeout(timeout):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _signal_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                return fn(*args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
        return wrapper
    return decorator


@pytest.mark.parametrize("protocol_classes", 
                         [[k] for k in _transport_protocols_registry.keys()], ids=lambda x: ",".join(x))
@timeout(1)
def test_protocol_wait_for_worker_timeout(protocol_classes):
    from nerfbaselines.backends._rpc import AutoTransportProtocol

    protocol_host = AutoTransportProtocol(protocol_classes=protocol_classes)
    protocol_host.start_host()
    with pytest.raises(TimeoutError):
        protocol_host.wait_for_worker(timeout=0.1)


@pytest.fixture()
def with_echo_protocol():
    def worker(cls, config):
        protocol_worker = cls(**config)
        protocol_worker.connect_worker()

        while True:
            data = protocol_worker.receive()
            if data.get("_end"):
                break
            protocol_worker.send(data)

    @contextlib.contextmanager
    def context(protocol_host):
        protocol_host.start_host()
        worker_thread = threading.Thread(
            target=worker, 
            args=(protocol_host.__class__, protocol_host.get_worker_configuration(),), 
            daemon=True)
        try:
            worker_thread.start()
            protocol_host.wait_for_worker(timeout=0.1)

            yield protocol_host

        finally:
            protocol_host.send({"_end": True})
            worker_thread.join()
    return context


@pytest.mark.parametrize("protocol_classes", 
                         [[k] for k in _transport_protocols_registry.keys()], ids=lambda x: ",".join(x))
@timeout(1)
def test_protocol_send_receive(protocol_classes, with_echo_protocol):
    from nerfbaselines.backends._rpc import AutoTransportProtocol
    import numpy as np

    with with_echo_protocol(AutoTransportProtocol(protocol_classes=protocol_classes)) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})
        out = echo_protocol.receive()
        assert np.array_equal(dummy_data, out["data"])


@timeout(1)
def test_protocol_shm_pickle_large_message(with_echo_protocol):
    from nerfbaselines.backends.protocol_shm_pickle import SharedMemoryProtocol
    import numpy as np

    with with_echo_protocol(SharedMemoryProtocol(shared_memory_size=100)) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})
        out = echo_protocol.receive()
        assert np.array_equal(dummy_data, out["data"])
