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
from multiprocessing import resource_tracker
try:
    from multiprocessing import shared_memory
except ImportError:
    shared_memory = None


DEFAULT_SHM_SIZE = 1080*1920*(12+12+12)


def _noop(*args, **kwargs): del args, kwargs


def _tcp_generate_authkey():
    return secrets.token_hex(64).encode("ascii")


def _protocol_defaults():
    protocol_type = "auto"
    shm_size = DEFAULT_SHM_SIZE
    pickle_protocol = pickle.HIGHEST_PROTOCOL
    hostname = "localhost"
    port = 0

    env_protocol = os.environ.get("NERFBASELINES_PROTOCOL")
    if env_protocol is not None:
        parts = os.environ.get("NERFBASELINES_PROTOCOL").split("-")
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


def _tcp_pickle_recv(conn: socket.socket, shm_buffer=None, zero_copy=False):
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
            buffer = memoryview(shm_buffer)[shm_offset:shm_offset+buffer_size]
            # Perform copy here? (for zero_copy=False)
            # NOTE: In the zero_copy mode, the buffer is not copied
            # The data are only valid until the next send/recv call
            if not zero_copy:
                buffer = buffer.tobytes()
            buffers.append(buffer)
    with io.BytesIO(pickle_bytes) as buf:
        return pickle.load(buf, buffers=buffers)


# def _align_page(offset):
#     a = mmap.PAGESIZE
#     return (offset + a - 1) & ~(a - 1)


def _tcp_pickle_send(conn: socket.socket, message, 
                     *,
                     pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
                     shm_buffer=None):
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
        shm_offset = 0
        for buffer in buffers:
            if shm_offset + buffer.nbytes > shm_buffer.nbytes:
                # We will make it network buffer
                header.append(2**64-1)
                header.append(buffer.nbytes)
                network_buffers.append(buffer)
            else:
                # We will copy data to shared memory
                header.append(shm_offset)
                header.append(buffer.nbytes)
                shm_buffer[shm_offset:shm_offset+buffer.nbytes] = buffer
                shm_offset += buffer.nbytes
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
        print(listeners)

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

    @property
    def protocol_name(self):
        if not self._conns:
            return 'not-connected'
        is_tcp = self._conns[0].family == socket.AF_INET
        base = "tcp" if is_tcp else "pipe"
        protocol_name = f"{base}-pickle{self._pickle_protocol}"
        if self._shm_size > 0:
            shm_size_str = _format_size(self._shm_size)
            protocol_name += f"-shm{shm_size_str}"
        return protocol_name

    def send(self, message, channel=0):
        assert self._is_host is not None, "Not started as host or worker"
        assert self._conns is not None, "Not connected"
        try:
            conn = self._conns[channel]
            shm_buffer = None
            if channel == 0 and self._shm is not None:
                shm_buffer = self._shm.buf
            _tcp_pickle_send(conn, message, pickle_protocol=self._pickle_protocol, shm_buffer=shm_buffer)
        except (EOFError, BrokenPipeError, ConnectionError) as e:
            raise ConnectionError("Connection error") from e

    def receive(self, channel=None, zero_copy=False):
        assert self._is_host is not None, "Not started as host or worker"
        assert self._conns is not None, "Not initialized"
        try:
            while True:
                if channel is None:
                    active_conns, _, _ = select.select(self._conns, [], [], None)
                    if active_conns:
                        conn = active_conns[-1]
                else:
                    conn = self._conns[channel]
                channel = self._conns.index(conn)
                shm_buffer = None
                if channel == 0 and self._shm is not None:
                    shm_buffer = self._shm.buf
                message = _tcp_pickle_recv(conn, shm_buffer=shm_buffer, zero_copy=zero_copy)
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
            except OSError:
                pass
            self._shm = None


def _connect_with_timeout(conn, *args, timeout=None, partial_timeout=5):
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
