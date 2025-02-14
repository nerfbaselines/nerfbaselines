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
                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                 num_channels=2):
        self._hostname = hostname
        self._port = port
        self._authkey = authkey

        self._is_host = None
        self._connected = False
        self._num_channels = num_channels
        self._conns = None
        self._tcp_listener = None
        self._shm = None
        self._shm_name = shm_name
        self._shm_size = shm_size
        self._pickle_protocol = min(pickle_protocol, pickle.HIGHEST_PROTOCOL)

    def start_host(self):
        if self._hostname is None:
            self._hostname = os.environ.get("NERFBASELINES_TCP_HOSTNAME", "localhost")
        if self._port is None:
            self._port = int(os.environ.get("NERFBASELINES_TCP_PORT", 0))
        if self._authkey is None:
            self._authkey = _tcp_generate_authkey()
        if self._shm_size is None:
            self._shm_size = int(os.environ.get("NERFBASELINES_SHM_SIZE", DEFAULT_SHM_SIZE))
            if shared_memory is None:
                logging.error("Shared memory is not available")
        self._is_host = True
        self._tcp_listener = socket.socket()
        self._tcp_listener.bind((self._hostname, self._port))
        self._port = self._tcp_listener.getsockname()[1]
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

        with self._tcp_listener as tcp_listener:
            tcp_listener.listen(self._num_channels)

            # Accept main connection
            listeners, _, _ = select.select([tcp_listener], [], [], timeout)
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

            # Release the listener
            self._tcp_listener = None

    def get_worker_configuration(self):
        out = {
            "authkey": self._authkey,
            "shm_name": self._shm_name,
            "shm_size": self._shm_size,
            "num_channels": self._num_channels,
            "pickle_protocol": self._pickle_protocol,
        }
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
        sock_args = (socket.AF_INET, socket.SOCK_STREAM)
        sockname = (self._hostname, self._port)
        conn = socket.socket(*sock_args)
        _connect_with_timeout(conn, sockname, timeout=timeout)
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
        protocol_name = f"tcp-pickle{self._pickle_protocol}"
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
        if self._is_host is None:
            return

        for x in (self._conns or []):
            x.close()
        self._conns = None
        if self._shm is not None:
            self._shm.unlink()
            self._shm.close()
            self._shm = None


def _connect_with_timeout(conn, *args, timeout=None):
    start = time.time()
    if timeout is not None:
        while time.time() - start < timeout:
            try:
                conn.settimeout(min(5, timeout))
                conn.connect(*args)
                break
            except ConnectionRefusedError:
                continue
            except socket.timeout:
                continue
        else:
            raise TimeoutError("Timeout waiting for connection")
        conn.setblocking(True)
    else:
        conn.setblocking(True)
        conn.connect(*args)


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
