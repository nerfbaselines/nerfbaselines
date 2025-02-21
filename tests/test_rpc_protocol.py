import time
import sys
import contextlib
import threading
import pytest
import functools
import signal


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


@timeout(4)
def test_protocol_wait_for_worker_timeout():
    from nerfbaselines.backends._transport_protocol import TransportProtocol as MessageProtocol
    protocol_host = MessageProtocol()
    try:
        protocol_host.start_host()
        with pytest.raises(TimeoutError):
            protocol_host.wait_for_worker(timeout=0.1)
    finally:
        protocol_host.close()


@pytest.fixture()
def with_echo_protocol():
    exc = []
    wait = [True]
    def worker(cls, config, host_first):
        if host_first:
            time.sleep(0.05)
        protocol_worker = None
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
        finally:
            try:
                if protocol_worker is not None:
                    protocol_worker.close()
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
            time.sleep(0.05)
            import gc
            gc.collect()
            protocol_host.close()
            worker_thread.join()

    context.exceptions = exc  # type: ignore
    return context


@timeout(4)
def test_protocol_send_receive(with_echo_protocol):
    from nerfbaselines.backends._transport_protocol import TransportProtocol as MessageProtocol
    import numpy as np

    with with_echo_protocol(MessageProtocol()) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        dummy_data2 = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data, "data2": dummy_data2})
        out = echo_protocol.receive()
        assert np.array_equal(dummy_data, out["data"])
        assert np.array_equal(dummy_data2, out["data2"])
        del out


@timeout(4)
def test_protocol_large_message(with_echo_protocol):
    from nerfbaselines.backends._transport_protocol import TransportProtocol
    from nerfbaselines.backends._common import backend_allocate_ndarray, set_allocator
    import numpy as np

    with with_echo_protocol(TransportProtocol()) as echo_protocol:
        # Simple message
        dummy_data = np.random.rand(100, 100)
        dummy_data2 = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data, "data2": dummy_data2})
        out = echo_protocol.receive()
        assert np.array_equal(dummy_data, out["data"])
        del out

    with with_echo_protocol(TransportProtocol()) as echo_protocol:
        # Test communication using allocator
        with set_allocator(echo_protocol.get_allocator()):
            dummy_data = np.random.rand(100, 100)
            dummy_data2 = np.random.rand(100, 100)
            _dummy_data2 = backend_allocate_ndarray(dummy_data2.shape, dummy_data2.dtype)
            np.copyto(_dummy_data2, dummy_data2)
            dummy_data3 = np.random.rand(100, 100)
            _dummy_data3 = backend_allocate_ndarray(dummy_data3.shape, dummy_data3.dtype)
            np.copyto(_dummy_data3, dummy_data3)
            dummy_data4 = np.random.rand(100, 100)
            echo_protocol.send({"data":[dummy_data, _dummy_data2, _dummy_data3, dummy_data4]})
            out = echo_protocol.receive()["data"]
            assert len(out) == 4
            assert np.array_equal(dummy_data, out[0])
            assert np.array_equal(dummy_data2, out[1])
            assert np.array_equal(dummy_data3, out[2])
            assert np.array_equal(dummy_data4, out[3])
            del out

    with with_echo_protocol(TransportProtocol()) as echo_protocol:
        # Test communication using allocator and zero_copy
        with set_allocator(echo_protocol.get_allocator()):
            dummy_data = np.random.rand(100, 100)
            dummy_data2 = np.random.rand(100, 100)
            _dummy_data2 = backend_allocate_ndarray(dummy_data2.shape, dummy_data2.dtype)
            np.copyto(_dummy_data2, dummy_data2)
            dummy_data3 = np.random.rand(100, 100)
            _dummy_data3 = backend_allocate_ndarray(dummy_data3.shape, dummy_data3.dtype)
            np.copyto(_dummy_data3, dummy_data3)
            dummy_data4 = np.random.rand(100, 100)
            echo_protocol.send({"data":[dummy_data, _dummy_data2, _dummy_data3, dummy_data4]})
            out = echo_protocol.receive(zero_copy=True)["data"]
            assert len(out) == 4
            assert np.array_equal(dummy_data, out[0])
            assert np.array_equal(dummy_data2, out[1])
            assert np.array_equal(dummy_data3, out[2])
            assert np.array_equal(dummy_data4, out[3])
            del out


@timeout(4)
def test_protocol_close_connection_host(with_echo_protocol):
    from nerfbaselines.backends._transport_protocol import TransportProtocol as MessageProtocol
    import numpy as np

    # Note, if the echo protocol thread wasn't killed, 
    # the context would not exit and the function would
    # timeout.

    with with_echo_protocol(MessageProtocol()) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})
        _ = echo_protocol.receive()
        del _

        # Test after receive
        echo_protocol.close()

    with with_echo_protocol(MessageProtocol()) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})

        # Test before receive
        echo_protocol.close()


@timeout(8)
def test_protocol_close_connection_worker(with_echo_protocol):
    from nerfbaselines.backends._transport_protocol import TransportProtocol as MessageProtocol

    with with_echo_protocol(MessageProtocol()) as echo_protocol:
        echo_protocol.send({"_action": "end_after_send"})
        echo_protocol.receive()

    with with_echo_protocol(MessageProtocol()) as echo_protocol:
        echo_protocol.send({"_action": "end_after_send"})
        echo_protocol.receive()

        # The connection should be broken at this point
        with pytest.raises(ConnectionError):
            echo_protocol.send({})
            echo_protocol.receive()

    with with_echo_protocol(MessageProtocol()) as echo_protocol:
        with pytest.raises(ConnectionError):
            echo_protocol.send({"_action": "end_after_receive"})
            echo_protocol.receive()


@timeout(4)
def test_protocol_establish_host_first(with_echo_protocol):
    from nerfbaselines.backends._transport_protocol import TransportProtocol as MessageProtocol
    import numpy as np

    with with_echo_protocol(MessageProtocol(), host_first=True) as echo_protocol:
        dummy_data = np.random.rand(100, 100)
        echo_protocol.send({"data": dummy_data})
        out = echo_protocol.receive()
        assert np.array_equal(dummy_data, out["data"])
        del out
