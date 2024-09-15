import time
import sys
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
            if not sys.platform.startswith("win"):
                signal.signal(signal.SIGALRM, _signal_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                return fn(*args, **kwargs)
            finally:
                if not sys.platform.startswith("win"):
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, signal.SIG_DFL)
        return wrapper
    return decorator


@pytest.mark.parametrize("protocol_classes", 
                         [[k] for k in _transport_protocols_registry.keys()], ids=lambda x: ",".join(x))
@timeout(4)
def test_protocol_wait_for_worker_timeout(protocol_classes):
    if "shm-pickle" in protocol_classes and sys.version_info < (3, 8):
        pytest.skip("Shared memory is not supported for Python < 3.8.")
    from nerfbaselines.backends._rpc import AutoTransportProtocol

    protocol_host = AutoTransportProtocol(protocol_classes=protocol_classes)
    protocol_host.start_host()
    with pytest.raises(TimeoutError):
        protocol_host.wait_for_worker(timeout=0.1)


@pytest.fixture()
def with_echo_protocol():
    exc = []
    wait = [True]
    def worker(cls, config, host_first):
        if host_first:
            time.sleep(0.05)
        try:
            protocol_worker = cls(**config)
            wait[0] = False
            protocol_worker.connect_worker()

            while True:
                data = protocol_worker.receive()
                if data.get("_end"):
                    break
                if data.get("_action") == "end_after_receive":
                    time.sleep(0.05)
                    protocol_worker.close()
                    break
                if data.get("_action") == "end_after_send":
                    protocol_worker.send({})
                    time.sleep(0.05)
                    protocol_worker.close()
                    break
                protocol_worker.send(data)
        except BaseException as e:
            exc.append(e)

    @contextlib.contextmanager
    def context(protocol_host, host_first=False):
        protocol_host.start_host()
        worker_thread = threading.Thread(
            target=worker, 
            args=(protocol_host.__class__, protocol_host.get_worker_configuration(), host_first), 
            daemon=True)
        try:
            worker_thread.start()
            if not host_first:
                while wait[0]:
                    time.sleep(0.005)
                time.sleep(0.05)
            protocol_host.wait_for_worker()
            yield protocol_host

        finally:
            protocol_host.close()
            worker_thread.join()

    context.exceptions = exc  # type: ignore
    return context


@pytest.mark.parametrize("protocol_classes", 
                         [[k] for k in _transport_protocols_registry.keys()], ids=lambda x: ",".join(x))
@timeout(4)
def test_protocol_send_receive(protocol_classes, with_echo_protocol):
    if "shm-pickle" in protocol_classes and sys.version_info < (3, 8):
        pytest.skip("Shared memory is not supported for Python < 3.8.")
    from nerfbaselines.backends._rpc import AutoTransportProtocol
    import numpy as np

    with with_echo_protocol(AutoTransportProtocol(protocol_classes=protocol_classes)) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})
        out = echo_protocol.receive()
        assert np.array_equal(dummy_data, out["data"])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
@timeout(4)
def test_protocol_shm_pickle_large_message(with_echo_protocol):
    from nerfbaselines.backends.protocol_shm_pickle import SharedMemoryProtocol
    import numpy as np

    with with_echo_protocol(SharedMemoryProtocol(shared_memory_size=128)) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})
        out = echo_protocol.receive()
        assert np.array_equal(dummy_data, out["data"])


@timeout(4)
def test_protocol_tcp_pickle_large_message(with_echo_protocol):
    from nerfbaselines.backends.protocol_tcp_pickle import TCPPickleProtocol
    import numpy as np

    with with_echo_protocol(TCPPickleProtocol(max_message_size=128)) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})
        out = echo_protocol.receive()
        assert np.array_equal(dummy_data, out["data"])


@pytest.mark.parametrize("protocol_classes", 
                         [[k] for k in _transport_protocols_registry.keys()] + [
                             ["tcp-pickle", "shm-pickle"]
                        ], ids=lambda x: ",".join(x))
@timeout(4)
def test_protocol_close_connection_host(protocol_classes, with_echo_protocol):
    if "shm-pickle" in protocol_classes and sys.version_info < (3, 8):
        pytest.skip("Shared memory is not supported for Python < 3.8.")
    from nerfbaselines.backends._rpc import AutoTransportProtocol
    import numpy as np

    # Note, if the echo protocol thread wasn't killed, 
    # the context would not exit and the function would
    # timeout.

    with with_echo_protocol(AutoTransportProtocol(protocol_classes=protocol_classes)) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})
        _ = echo_protocol.receive()

        # Test after receive
        echo_protocol.close()

    with with_echo_protocol(AutoTransportProtocol(protocol_classes=protocol_classes)) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})

        # Test before receive
        echo_protocol.close()


@pytest.mark.parametrize("protocol_classes", 
                         [[k] for k in _transport_protocols_registry.keys()] + [
                             ["tcp-pickle", "shm-pickle"]
                        ], ids=lambda x: ",".join(x))
@timeout(8)
def test_protocol_close_connection_worker(protocol_classes, with_echo_protocol):
    if "shm-pickle" in protocol_classes and sys.version_info < (3, 8):
        pytest.skip("Shared memory is not supported for Python < 3.8.")
    from nerfbaselines.backends._rpc import AutoTransportProtocol

    with with_echo_protocol(AutoTransportProtocol(protocol_classes=protocol_classes)) as echo_protocol:
        echo_protocol.send({"_action": "end_after_send"})
        echo_protocol.receive()

    with with_echo_protocol(AutoTransportProtocol(protocol_classes=protocol_classes)) as echo_protocol:
        echo_protocol.send({"_action": "end_after_send"})
        echo_protocol.receive()

        # The connection should be broken at this point
        with pytest.raises(ConnectionError):
            echo_protocol.send({})
            echo_protocol.receive()

    with with_echo_protocol(AutoTransportProtocol(protocol_classes=protocol_classes)) as echo_protocol:
        with pytest.raises(ConnectionError):
            echo_protocol.send({"_action": "end_after_receive"})
            echo_protocol.receive()


@pytest.mark.parametrize("protocol_classes", 
                         [[k] for k in _transport_protocols_registry.keys()] + [
                             ["tcp-pickle", "shm-pickle"]
                        ], ids=lambda x: ",".join(x))
@timeout(4)
def test_protocol_establish_host_first(protocol_classes, with_echo_protocol):
    if "shm-pickle" in protocol_classes and sys.version_info < (3, 8):
        pytest.skip("Shared memory is not supported for Python < 3.8.")
    from nerfbaselines.backends._rpc import AutoTransportProtocol
    import numpy as np

    with with_echo_protocol(AutoTransportProtocol(protocol_classes=protocol_classes), host_first=True) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})
        out = echo_protocol.receive()
        assert np.array_equal(dummy_data, out["data"])
