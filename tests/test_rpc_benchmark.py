import sys
import pytest
import struct
import numpy as np
import pickle

MESSAGE_SIZE = 64 * 1024 * 1024  # 128 MB
PORT = 8234


def _limit_message_size(message, message_size=MESSAGE_SIZE):
    if len(message) < message_size:
        yield message
    else:
        for i in range(0, len(message), message_size):
            lastm = message[i : i + message_size]
            yield lastm
        if len(lastm) == message_size:  # type: ignore
            yield b""


def _collect_message_parts(message, message_size=MESSAGE_SIZE):
    m = next(message)
    parts = [m]
    while len(m) == message_size:
        m = next(message)
        parts.append(m)
    if len(parts) == 1:
        return parts[0]
    return b"".join(parts)


def serialize1(message):
    yield pickle.dumps(message)

def deserialize1(message):
    message = next(message)
    return pickle.loads(message)


def serialize2(message):
    buffers = []
    serialized = pickle.dumps(message, protocol=5, buffer_callback=lambda buffer: buffers.append(buffer))
    result = struct.pack("!i", len(buffers) + 1)
    result += struct.pack("!i", len(serialized))
    for b in buffers:
        result += struct.pack("!i", len(b.raw()))
    yield result + b''.join([serialized] + buffers)

def deserialize2(message):
    message = next(message)
    num_buffers, = struct.unpack("!I", message[:4])
    offset = 4*(num_buffers+1)
    buff_lens = struct.unpack("!"+"I"*num_buffers, message[4:offset])
    buffers = []
    for buff_len in buff_lens:
        buffers.append(message[offset:offset + buff_len])
        offset += buff_len
    return pickle.loads(buffers[0], buffers=buffers[1:])


def serialize3(message):
    buffers = []
    serialized = pickle.dumps(message, protocol=5, buffer_callback=lambda buffer: buffers.append(buffer))
    yield from _limit_message_size(struct.pack("!i", len(buffers)) + serialized)
    for buffer in buffers:
        yield from _limit_message_size(buffer.raw())

def deserialize3(message):
    header = _collect_message_parts(message)
    num_buffers, = struct.unpack("!I", header[:4])
    buffers = []
    for _ in range(num_buffers):
        buffers.append(_collect_message_parts(message))
    return pickle.loads(header[4:], buffers=buffers)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
@pytest.mark.benchmark(group="rpc-transport")
@pytest.mark.parametrize("serialize, deserialize", [
    (serialize1, deserialize1), 
    (serialize2, deserialize2), 
    (serialize3, deserialize3)], ids=["pickle", "pickle5", "pickle5+buffers"])
@pytest.mark.parametrize("image_dtype", [np.uint8, np.float32])
def test_http_transport(benchmark, serialize, deserialize, image_dtype):
    message = {
        "color": np.random.normal(size=(1920, 1080, 3)).astype(image_dtype),
        "depth": np.random.normal(size=(1920, 1080, 1)).astype(np.float32),
    }

    from multiprocessing.connection import Listener, Client
    import threading

    # Benchmark serialize and deserialize when sent through a Listener, Client connection
    with Listener(address=("localhost", PORT), authkey=b'secret password') as listener:
        def handle_client(listener):
            with listener.accept() as conn:
                while conn.recv():
                    for m in serialize(message):
                        conn.send_bytes(m)
        threading.Thread(target=handle_client, args=(listener,)).start()

        with Client(address=("localhost", PORT), authkey=b'secret password') as client:
            def _measure():
                client.send(True)
                def _iter_response():
                    while True:
                        yield client.recv_bytes()
                return deserialize(_iter_response())
            benchmark(_measure)
            client.send(False)


# Skip on python < 3.8
@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
@pytest.mark.benchmark(group="rpc-transport")
@pytest.mark.parametrize("serialize, deserialize", [
    (serialize1, deserialize1), 
    (serialize2, deserialize2), 
    (serialize3, deserialize3)], ids=["pickle", "pickle5", "pickle5+buffers"])
@pytest.mark.parametrize("image_dtype", [np.uint8, np.float32])
def test_shared_memory_transport(benchmark, serialize, deserialize, image_dtype):
    import multiprocessing.shared_memory
    message = {
        "color": np.random.normal(size=(1920, 1080, 3)).astype(image_dtype),
        "depth": np.random.normal(size=(1920, 1080, 1)).astype(np.float32),
    }

    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=MESSAGE_SIZE)

    from multiprocessing.connection import Listener, Client
    import threading

    try:

        # Benchmark serialize and deserialize when sent through a Listener, Client connection
        with Listener(address=("localhost", PORT), authkey=b'secret password') as listener:
            def handle_client(listener):
                shm_local = multiprocessing.shared_memory.SharedMemory(name=shm.name)
                try:
                    with listener.accept() as conn:
                        while conn.recv():
                            for i, m in enumerate(serialize(message)):
                                shm_local.buf[:len(m)] = m
                                conn.send_bytes(struct.pack("!ii", i, len(m)))
                                conn.recv_bytes()
                finally:
                    shm_local.close()
            threading.Thread(target=handle_client, args=(listener,)).start()

            with Client(address=("localhost", PORT), authkey=b'secret password') as client:
                def _measure():
                    client.send(True)
                    def _iter_response():
                        while True:
                            i, l = struct.unpack("!II", client.recv_bytes())
                            del i
                            m = shm.buf[:l].tobytes()
                            client.send_bytes(b"")
                            yield m
                    return deserialize(_iter_response())
                benchmark(_measure)
                client.send(False)
    finally:
        shm.close()
        shm.unlink()


# Skip on python < 3.8
@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires python3.8 or higher")
@pytest.mark.benchmark(group="rpc-transport")
@pytest.mark.parametrize("serialize, deserialize", [
    (serialize3, deserialize3)], ids=["pickle5+buffers"])
@pytest.mark.parametrize("image_dtype", [np.uint8, np.float32])
def test_shm_merge_messages(benchmark, serialize, deserialize, image_dtype):
    import multiprocessing.shared_memory
    from multiprocessing.connection import Listener, Client

    message = {
        "color": np.random.normal(size=(1920, 1080, 3)).astype(image_dtype),
        "depth": np.random.normal(size=(1920, 1080, 1)).astype(np.float32),
    }

    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=MESSAGE_SIZE)
    import threading

    try:

        # Benchmark serialize and deserialize when sent through a Listener, Client connection
        with Listener(address=("localhost", PORT), authkey=b'secret password') as listener:
            def handle_client(listener):
                shm_local = multiprocessing.shared_memory.SharedMemory(name=shm.name)
                try:
                    with listener.accept() as conn:
                        while conn.recv():
                            for i, m in enumerate(serialize(message)):
                                shm_local.buf[:len(m)] = m
                                conn.send_bytes(struct.pack("!ii", i, len(m)))
                                conn.recv_bytes()
                finally:
                    shm_local.close()
            threading.Thread(target=handle_client, args=(listener,)).start()

            with Client(address=("localhost", PORT), authkey=b'secret password') as client:
                def _measure():
                    client.send(True)
                    def _iter_response():
                        while True:
                            i, l = struct.unpack("!II", client.recv_bytes())
                            del i
                            m = shm.buf[:l].tobytes()
                            client.send_bytes(b"")
                            yield m
                    return deserialize(_iter_response())
                benchmark(_measure)
                client.send(False)
    finally:
        shm.close()
        shm.unlink()
