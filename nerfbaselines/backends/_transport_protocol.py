import ctypes
import functools
import weakref
import stat
import sys
import tempfile
import logging
import io
import select
import struct
import os
import time
import pickle
import socket
import secrets
import numpy as np
try:
    from multiprocessing import shared_memory
except ImportError:
    shared_memory = None


DEFAULT_SHM_SIZE = 1080*1920*(12+12+12)


def _noop(*args, **kwargs): del args, kwargs


def _tcp_generate_authkey():
    return secrets.token_hex(64).encode("ascii")


class _allocator:
    def __init__(self, buffer):
        self._offset = 0
        if buffer is not None:
            self._buffer = weakref.ref(buffer)
        else:
            self._buffer = _noop
    
    def allocate(self, size):
        buffer = self._buffer()
        if buffer is None:
            return None
        offset = self._offset
        if offset + size > buffer.nbytes:
            return None
        out = buffer[offset:offset+size]
        self._offset += size
        return offset, out

    def allocate_ndarray(self, shape, dtype):
        nelem = functools.reduce(lambda x, y: x * y, shape, 1)
        nbytes = nelem * np.dtype(dtype).itemsize
        buffer = self._buffer()
        allocation = self.allocate(nbytes)
        if allocation is None or buffer is None:
            return np.ndarray(shape, dtype=dtype)
        offset, _ = allocation
        return np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)

    def get_allocation_offset(self, buffer):
        self_buffer = self._buffer()
        if self_buffer is None or buffer is None:
            return None
        if self_buffer.readonly or buffer.readonly:
            return None
        self_ptr = ctypes.addressof(ctypes.c_char.from_buffer(self_buffer))
        buffer_ptr = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        if self_ptr <= buffer_ptr < self_ptr + self_buffer.nbytes:
            return buffer_ptr - self_ptr
        return None

    def get(self, offset, size):
        buffer = self._buffer()
        assert buffer is not None, "Failed to access shared memory"
        return buffer[offset:offset+size]

    def reset(self):
        self._offset = 0


def _protocol_defaults():
    protocol_type = "auto"
    shm_size = DEFAULT_SHM_SIZE
    pickle_protocol = pickle.HIGHEST_PROTOCOL
    hostname = "localhost"
    port = 0

    env_protocol = os.environ.get("NERFBASELINES_PROTOCOL")
    if env_protocol is not None:
        parts = env_protocol.split("-")
        protocol_type = parts[0]
        if protocol_type not in ("tcp", "pipe", "auto"):
            raise ValueError(f"Unsupported protocol type {protocol_type} "
                f"in NERFBASELINES_PROTOCOL={env_protocol}, expected one of 'tcp', 'pipe', 'auto'")
        parts = parts[1:]
        for part in parts:
            if part.startswith("shm"):
                try:
                    shm_size_str = part[3:].lower()
                    if shm_size_str.endswith("k"):
                        shm_size = int(shm_size_str[:-1]) * 1024
                    elif shm_size_str.endswith("m"):
                        shm_size = int(shm_size_str[:-1]) * 1024 * 1024
                    elif shm_size_str.endswith("g"):
                        shm_size = int(shm_size_str[:-1]) * 1024 * 1024 * 1024
                    else:
                        shm_size = int(shm_size_str)
                except ValueError:
                    raise ValueError(f"Invalid shared memory size part {part} in NERFBASELINES_PROTOCOL={env_protocol}")
            elif part.startswith("pickle"):
                pickle_protocol = int(part[6:])
            else:
                raise ValueError(f"Unsupported protocol part {part} in NERFBASELINES_PROTOCOL={env_protocol}")
    env_hostname = os.environ.get("NERFBASELINES_TCP_HOSTNAME")
    if env_hostname is not None:
        hostname = env_hostname
    env_port = os.environ.get("NERFBASELINES_TCP_PORT")
    if env_port is not None:
        port = int(env_port)
    env_shm_size = os.environ.get("NERFBASELINES_SHM_SIZE")
    if env_shm_size is not None:
        shm_size = int(env_shm_size)
    return protocol_type, shm_size, pickle_protocol, hostname, port


def _format_size(size):
    unit = ""
    if size > 1024 * 10:
        size //= 1024
        unit = "K"
    if size > 1024 * 10:
        size //= 1024
        unit = "M"
    if size > 1024 * 10:
        size //= 1024
        unit = "G"
    return f"{size}{unit}"

def _socket_exists(path):
    if os.path.exists(path):
        st = os.stat(path)
        if stat.S_ISSOCK(st.st_mode):
            return True
    return False


def _tcp_pickle_recv(conn: socket.socket, allocator=None, zero_copy=False):
    def _read_buffer(size):
        buffer = bytearray(size)
        i = 0
        while i < size:
            n = conn.recv_into(memoryview(buffer)[i:], size-i)
            if n == 0:
                if i == size: raise EOFError
                else: raise OSError("got end of file during message")
            i += n
        return buffer

    data = conn.recv(12)
    if len(data) < 12: raise EOFError
    num_buffers, size, = struct.unpack("!iQ", data)
    header = struct.unpack(f"!{(num_buffers-1)*2}Q", conn.recv(16*(num_buffers-1)))
    pickle_bytes = _read_buffer(size)
    buffers = []
    for shm_offset, buffer_size in zip(header[::2], header[1::2]):
        if shm_offset == 2**64-1:
            # Network buffer
            buffers.append(_read_buffer(buffer_size))
        else:
            # Shared memory buffer
            assert allocator is not None, "Shared memory buffer without allocator"
            buffer = allocator.get(shm_offset, buffer_size)
            # Perform copy here? (for zero_copy=False)
            # NOTE: In the zero_copy mode, the buffer is not copied
            # The data are only valid until the next send/recv call
            if not zero_copy:
                buffer = bytearray(buffer)
            buffers.append(buffer)
    with io.BytesIO(pickle_bytes) as buf:
        if buffers:
            return pickle.load(buf, buffers=buffers)
        else:
            return pickle.load(buf)


# def _align_page(offset):
#     a = mmap.PAGESIZE
#     return (offset + a - 1) & ~(a - 1)


def _tcp_pickle_send(conn: socket.socket, message, 
                     *,
                     pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
                     allocator=None):
    buffers = []
    def buffer_callback(buffer):
        size = buffer.raw().nbytes
        # Don't use buffers for small objects
        if size < 256: return True
        buffers.append(buffer.raw())
    network_buffers = []
    with io.BytesIO() as buf:
        if pickle_protocol >= 5:
            pickle.dump(message, buf, protocol=pickle_protocol, buffer_callback=buffer_callback)
        else:
            pickle.dump(message, buf, protocol=pickle_protocol)
        size = buf.tell()
        header = [len(buffers)+1, size]
        buf.seek(0)
        for buffer in buffers:
            # Check if buffer already is in allocator's memory
            if allocator is not None:
                shm_offset = allocator.get_allocation_offset(buffer)
                if shm_offset is not None:
                    header.append(shm_offset)
                    header.append(buffer.nbytes)
                    continue

                # Try allocating buffer in the shared memory
                allocation = allocator.allocate(buffer.nbytes)
                if allocation is not None:
                    # We will copy data to shared memory
                    shm_offset, out_buffer = allocation
                    out_buffer[:] = buffer
                    header.append(shm_offset)
                    header.append(buffer.nbytes)
                    continue

            # We will make it network buffer
            header.append(2**64-1)
            network_buffers.append(buffer)
            header.append(buffer.nbytes)
        conn.sendall(struct.pack(f"!i{len(header)-1}Q", *header))
        conn.sendall(buf.getbuffer())
        for buffer in network_buffers:
            conn.sendall(buffer)


class TransportPickler(pickle.Pickler):
    ...


class TransportProtocol:
    def __init__(self,
                 *,
                 hostname=None,
                 port=None,
                 authkey=None,
                 shm_name=None,
                 shm_size=None,
                 pipe_name=None,
                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                 num_channels=2):
        self._hostname = hostname
        self._port = port
        self._authkey = authkey
        self._pipe_name = pipe_name

        self._is_host = None
        self._connected = False
        self._num_channels = num_channels
        self._conns = None
        self._tcp_listener = None
        self._pipe_listener = None
        self._shm = None
        self._shm_name = shm_name
        self._shm_size = shm_size
        self._pickle_protocol = min(pickle_protocol, pickle.HIGHEST_PROTOCOL)
        self._tmpdir = None
        self._allocator = _allocator(None)

    def start_host(self):
        assert self._pipe_listener is None and self._tcp_listener is None, "Already started"
        if self._authkey is None:
            self._authkey = _tcp_generate_authkey()
        protocol_type, shm_size, pickle_protocol, hostname, port = _protocol_defaults()
        self._pickle_protocol = min(min(self._pickle_protocol, pickle_protocol), pickle.HIGHEST_PROTOCOL)
        if self._hostname is None:
            self._hostname = hostname
        if self._port is None:
            self._port = port
        if self._shm_size is None:
            self._shm_size = shm_size
            if shared_memory is None:
                logging.error("Shared memory is not available")
        self._is_host = True
        if protocol_type in ("tcp", "auto"):
            self._tcp_listener = socket.socket()
            self._tcp_listener.bind((self._hostname, self._port))
            self._port = self._tcp_listener.getsockname()[1]

        # Setup pipe listener on Unix
        if protocol_type in ("pipe", "auto") and sys.platform != "win32":
            if self._tmpdir is None:
                self._tmpdir = tempfile.TemporaryDirectory()
            pipe_name = os.path.join(self._tmpdir.name, "pipe")
            self._pipe_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._pipe_listener.bind(pipe_name)

        if self._pipe_listener is None and self._tcp_listener is None:
            raise RuntimeError("No listener available, please check the configuration")

        # Setup shared memory
        if shared_memory is not None and self._shm_size > 0:
            self._shm = shared_memory.SharedMemory(create=True, size=self._shm_size)
            self._shm_name = self._shm.name
        else:
            self._shm_size = 0

    def _setup_protocol(self, conn):
        _tcp_pickle_send(conn, {
            "message": "ready",
            "configuration": self.get_worker_configuration(),
        }, pickle_protocol=1)
        msg = _tcp_pickle_recv(conn)
        assert msg["message"] == "ready_ack", f"Unexpected message {msg['message']}"

    def wait_for_worker(self, timeout=None):
        assert self._is_host is not None, "Not started as host or worker"

        listeners = []
        if self._pipe_listener is not None:
            self._pipe_listener.listen(self._num_channels)
            listeners.append(self._pipe_listener)
        if self._tcp_listener is not None:
            self._tcp_listener.listen(self._num_channels)
            listeners.append(self._tcp_listener)
        assert listeners, "No listeners available"

        # Accept main connection
        listeners, _, _ = select.select(listeners, [], [], timeout)
        if not listeners:
            raise TimeoutError("Timeout waiting for worker")
        listener = listeners[0]
        conn, _ = listener.accept()

        self._conns = [conn]
        conn.setblocking(True)
        answer_challenge(conn, self._authkey)
        deliver_challenge(conn, self._authkey)

        # Setup with safest pickle protocol (1) for backward compatibility
        setup_response = _tcp_pickle_recv(conn)
        if setup_response["message"] != "ready":
            raise RuntimeError(f"Unexpected message {setup_response['message']}")
        if self._shm is not None:
            # We can unlink the shared memory now
            self._shm.unlink()
            self._shm.unlink = lambda: None
        # Use the response to fix current configuration
        self._shm_size = setup_response["configuration"]["shm_size"]
        self._pickle_protocol = setup_response["configuration"]["pickle_protocol"]
        if self._shm_size <= 0 and self._shm is not None:
            # Release the shared memory
            self._shm.close()
            self._shm = None
        _tcp_pickle_send(conn, {"message": "ready_ack"}, pickle_protocol=1)

        # Accept additional connections
        for _ in range(self._num_channels - 1):
            self._conns.append(listener.accept()[0])

        # Setup the allocator
        if self._shm is not None:
            self._allocator = _allocator(self._shm.buf)

        # Release the listeners
        if self._tcp_listener is not None:
            self._tcp_listener.close()
            self._tcp_listener = None
        if self._pipe_listener is not None:
            self._pipe_listener.close()
            self._pipe_listener = None

    def get_worker_configuration(self):
        out = {
            "authkey": self._authkey,
            "shm_name": self._shm_name,
            "shm_size": self._shm_size,
            "num_channels": self._num_channels,
            "pickle_protocol": self._pickle_protocol,
        }
        if self._pipe_listener is not None:
            out["pipe_name"] = self._pipe_listener.getsockname()
        if self._tcp_listener is not None:
            out["hostname"], out["port"] = self._tcp_listener.getsockname()
        return out

    def _worker_try_setup_shm(self):
        # Try setup shared memory
        if self._shm_name is not None and shared_memory is not None:
            # Remove tracked shared memory as it is already tracked in the main thread
            from multiprocessing import resource_tracker
            old_register = resource_tracker.register
            try:
                resource_tracker.register = _noop
                self._shm = shared_memory.SharedMemory(name=self._shm_name, create=False)
                # Will be unlinked in the main thread
                self._shm.unlink = lambda: None
            except Exception as e:
                logging.error(f"Failed to connect to shared memory {self._shm_name}: {e}")
                self._shm_size = 0
                self._shm_name = None
            finally:
                resource_tracker.register = old_register
        else:
            self._shm_size = 0
            self._shm_name = None

    def connect_worker(self, timeout=None):
        # Try setup shared memory
        self._worker_try_setup_shm()
        self._is_host = False

        partial_timeout = 5
        if self._pipe_name is not None and _socket_exists(self._pipe_name):
            sock_args = (socket.AF_UNIX, socket.SOCK_STREAM)
            sockname = self._pipe_name
            partial_timeout = 0.5
        else:
            sock_args = (socket.AF_INET, socket.SOCK_STREAM)
            sockname = (self._hostname, self._port)
        conn = socket.socket(*sock_args)
        _connect_with_timeout(conn, sockname, timeout=timeout, partial_timeout=partial_timeout)
        deliver_challenge(conn, self._authkey)
        answer_challenge(conn, self._authkey)

        # Establish the protocol
        self._setup_protocol(conn)
        self._conns = [conn]

        # Connect additional channels
        for _ in range(self._num_channels - 1):
            conn = socket.socket(*sock_args)
            conn.connect(sockname)
            self._conns.append(conn)

        # Setup the allocator
        if self._shm is not None:
            self._allocator = _allocator(self._shm.buf)

    @property
    def protocol_name(self):
        if not self._conns:
            return 'not-connected'
        is_tcp = self._conns[0].family == socket.AF_INET
        base = "tcp" if is_tcp else "pipe"
        protocol_name = f"{base}-pickle{self._pickle_protocol}"
        if (self._shm_size or 0) > 0:
            shm_size_str = _format_size(self._shm_size)
            protocol_name += f"-shm{shm_size_str}"
        return protocol_name

    def send(self, message, channel=0):
        assert self._is_host is not None, "Not started as host or worker"
        assert self._conns is not None, "Not connected"
        try:
            conn = self._conns[channel]
            allocator = self.get_allocator(channel)
            _tcp_pickle_send(conn, message, pickle_protocol=self._pickle_protocol, allocator=allocator)
            allocator.reset()
        except (EOFError, BrokenPipeError, ConnectionError) as e:
            raise ConnectionError("Connection error") from e

    def receive(self, channel=None, zero_copy=False):
        assert self._is_host is not None, "Not started as host or worker"
        assert self._conns is not None, "Not initialized"
        try:
            while True:
                if channel is None:
                    active_conns, _, _ = select.select(self._conns, [], [], None)
                    if not active_conns:
                        continue
                    conn = active_conns[-1]
                else:
                    conn = self._conns[channel]
                channel = self._conns.index(conn)
                allocator = self.get_allocator(channel)
                message = _tcp_pickle_recv(conn, allocator=allocator, zero_copy=zero_copy)
                if isinstance(message, Exception):
                    raise message
                return message
        except ConnectionError:
            raise
        except (EOFError, BrokenPipeError) as e:
            raise ConnectionError(str(e)) from e
        except (OSError) as e:
            if "Bad file descriptor" in str(e):
                raise ConnectionError(str(e)) from e
            raise

    def get_allocator(self, channel=0):
        return self._allocator if channel == 0 else _allocator(None)

    def close(self):
        # Release the listeners
        if self._tcp_listener is not None:
            self._tcp_listener.close()
            self._tcp_listener = None
        if self._pipe_listener is not None:
            self._pipe_listener.close()
            self._pipe_listener = None

        # Release the connections
        for x in (self._conns or []):
            x.close()
        self._conns = None

        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

        # Release the shared memory
        if self._shm is not None:
            self._shm.unlink()
            try:
                self._shm.close()
            except (OSError, BufferError):
                pass
            self._shm = None


def _connect_with_timeout(conn, *args, timeout=None, partial_timeout: float = 5):
    start = time.time()
    while timeout is None or time.time() - start <= timeout:
        try:
            _timeout = partial_timeout
            if timeout is not None:
                _timeout = min(_timeout, timeout - (time.time() - start))
                if _timeout <= 0:
                    raise TimeoutError("Timeout waiting for connection")
            conn.settimeout(_timeout)
            conn.connect(*args)
            break
        except ConnectionRefusedError:
            continue
        except socket.timeout:
            continue
    else:
        raise TimeoutError("Timeout waiting for connection")
    conn.setblocking(True)


def deliver_challenge(conn: socket.socket, authkey):
    import hmac
    message = os.urandom(20)
    conn.sendall(message)
    digest = hmac.new(authkey, message, 'md5').digest()
    response = conn.recv(len(digest))
    if len(response) < len(digest):
        raise ConnectionError("Failed to receive response")
    conn.sendall(b'1' if response == digest else b'0')


def answer_challenge(conn: socket.socket, authkey):
    import hmac
    message = conn.recv(20)
    if len(message) != 20:
        raise ConnectionError("Failed to receive challenge")
    digest = hmac.new(authkey, message, 'md5').digest()
    conn.sendall(digest)
    response = conn.recv(1)
    if response != b'1':
        raise ConnectionError('Failed to authenticate')
