import random
import contextlib
import struct
import threading
import os
import time
import pickle
import socket
from typing import Optional
import secrets
import logging
from multiprocessing import connection as mp_connection
from multiprocessing.connection import Listener, Connection
from queue import Queue
from ..utils import CancellationToken


_TCP_MAX_MESSAGE_SIZE = 128 * 1024 * 1024  # 128 MB


def _tcp_generate_authkey():
    return secrets.token_hex(64).encode("ascii")


def _tcp_pickle_recv(connection: Connection):
    header = connection.recv_bytes()
    num_buffers, flags, max_block_size = struct.unpack("IIQ", header[:16])
    del flags
    buff_lens = struct.unpack("Q"*num_buffers, header[16:16+8*num_buffers])
    buffers = []
    for buff_len in buff_lens:
        # Allocate buffer
        buffer = bytearray(buff_len)
        buffers.append(buffer)
        
        for i in range(0, buff_len, max_block_size):
            mess_len = min(buff_len - i, max_block_size)
            assert connection.recv_bytes_into(memoryview(buffer)[i:i+mess_len]) == mess_len, "Failed to read message"
    return pickle.loads(buffers[-1], **({'buffers': buffers[:-1]} if len(buffers) > 1 else {}))


def _tcp_pickle_send(connection: Connection, message, 
          pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
          pickle_use_buffers: bool = True,
          max_http_message_size: int = _TCP_MAX_MESSAGE_SIZE):
    buffers = []
    def _add_buffer(buffer):
        buffers.append(buffer.raw())
    buffers.append(pickle.dumps(message, protocol=pickle_protocol,
       **({ "buffer_callback": _add_buffer } if (pickle_use_buffers and pickle_protocol >= 5) else {})))
    flags = 0
    header = struct.pack("IIQ", len(buffers), flags, max_http_message_size)
    header += struct.pack("Q"*len(buffers), *(len(buff) for buff in buffers))
    connection.send_bytes(header)
    for buffer in buffers:
        for i in range(0, len(buffer), max_http_message_size):
            mess_len = min(len(buffer) - i, max_http_message_size)
            connection.send_bytes(memoryview(buffer)[i:i+mess_len])





def _disable_reenter(f):
    inside_call = False
    def wrapper(*args, **kwargs):
        nonlocal inside_call
        if inside_call:
            raise RuntimeError("Re-entering the same function is not allowed.")
        inside_call = True
        try:
            return f(*args, **kwargs)
        finally:
            inside_call = False
        
    return wrapper


class TCPPickleProtocol:
    def __init__(self,
                 *,
                 hostname=None,
                 port=None,
                 authkey=None):
        if hostname is None:
            hostname = os.environ.get("NB_TCP_HOSTNAME", "localhost")
        if port is None:
            port = int(os.environ.get("NB_TCP_PORT", random.randint(10000, 20000)))
        if authkey is None:
            authkey = _tcp_generate_authkey()
        self._hostname = hostname
        self._port = port
        self._authkey = authkey

        self._is_host = None
        self._connected = False
        self._transport_options = {
            "pickle_protocol": 1,
            "pickle_use_buffers": False,
            "max_http_message_size": _TCP_MAX_MESSAGE_SIZE,
        }

        self._conn = None

        self._singlerun_contexts = {}

        # Worker specific:
        self._listener = None
        self._queue = None
        self._queue_interrupt = None
        self._lock = None
        self._receiver_thread = None

    @contextlib.contextmanager
    def _protect_singlerun(self, *args):
        if args in self._singlerun_contexts:
            raise RuntimeError("Re-entering the same function is not allowed.")
        self._singlerun_contexts[args] = True
        try:
            yield
        finally:
            del self._singlerun_contexts[args]

    def start_host(self):
        self._is_host = True

    def wait_for_worker(self, timeout=None):
        assert self._is_host is not None, "Not started as host or worker"

        self._conn = _client_connect_with_cancel(
            (self._hostname, self._port), self._authkey, None, 0 if timeout is None else timeout)

        # Establish the protocol
        msg = _tcp_pickle_recv(self._conn)
        assert msg["message"] == "ready", f"Unexpected message {msg['message']}"
        transport_options = msg["transport_options"]
        transport_options["pickle_protocol"] = min(
            transport_options["pickle_protocol"],
            pickle.HIGHEST_PROTOCOL)
        transport_options["pickle_use_buffers"] = (
            transport_options.get("pickle_use_buffers", False) and 
            transport_options["pickle_protocol"] >= 5)
        transport_options["max_http_message_size"] = min(
            transport_options["max_http_message_size"],
            _TCP_MAX_MESSAGE_SIZE)
        _tcp_pickle_send(self._conn, {
            "message": "ready_ack",
            "transport_options": transport_options,
        }, pickle_protocol=1, pickle_use_buffers=False)
        self._transport_options = transport_options

    def get_worker_configuration(self):
        assert self._is_host is True, "Not started as host"
        return {
            "hostname": self._hostname,
            "port": self._port,
            "authkey": self._authkey,
        }

    @staticmethod
    def _worker_get_messages_thread(queue, queue_interrupt, lock, conn: Connection):
        try:
            while not conn.closed:
                conn.poll(None)
                with lock:
                    msg = _tcp_pickle_recv(conn)
                if msg.get("__interrupt__", False):
                    queue_interrupt.put(msg)
                else:
                    queue.put(msg)
        except Exception as e:
            queue.put(e)
            queue_interrupt.put(e)

    def connect_worker(self):
        self._is_host = False

        # Initialize resources
        self._listener = Listener((self._hostname, self._port), authkey=self._authkey)

        # Accept the connection
        self._conn = self._listener.accept()
        self._queue = Queue()
        self._queue_interrupt = Queue()
        self._lock = threading.Lock()

        # Setup with safest pickle protocol (1) for backward compatibility
        _tcp_pickle_send(self._conn, {
            "message": "ready", 
            "transport_options": {
                "pickle_protocol": pickle.HIGHEST_PROTOCOL,
                "pickle_use_buffers": pickle.HIGHEST_PROTOCOL >= 5,
                "max_http_message_size": _TCP_MAX_MESSAGE_SIZE,
            },
        }, pickle_protocol=1, pickle_use_buffers=False)
        setup_response = _tcp_pickle_recv(self._conn)
        if setup_response["message"] != "ready_ack":
            raise RuntimeError(f"Unexpected message {setup_response['message']}")
        self._transport_options = setup_response["transport_options"]

        self._receiver_thread = threading.Thread(
             daemon=True, 
             target=self._worker_get_messages_thread,
             args=(self._queue, self._queue_interrupt, self._lock, self._conn))
        self._receiver_thread.start()

    @property
    def protocol_name(self):
        protocol_name = f"tcp-pickle{self._transport_options['pickle_protocol']}"
        if self._transport_options["pickle_use_buffers"]:
            protocol_name += "-buffers"
        return protocol_name

    def send(self, message, interrupt=False):
        assert self._is_host is not None, "Not started as host or worker"
        assert self._is_host or not interrupt, "Only host can send interrupt messages"
        with self._protect_singlerun(("send", interrupt)):
            if interrupt:
                message = {**message, "__interrupt__": True}
            if self._is_host:
                _tcp_pickle_send(self._conn, message, **self._transport_options)
            else:
                with self._lock:
                    _tcp_pickle_send(self._conn, message, **self._transport_options)

    def receive(self, interrupt=False):
        assert self._is_host is not None, "Not started as host or worker"
        assert not self._is_host or not interrupt, "Only worker can receive interrupt messages"
        with self._protect_singlerun(("receive", interrupt,)):
            if self._is_host:
                message = _tcp_pickle_recv(self._conn)
            else:
                if interrupt:
                    message = self._queue_interrupt.get()
                else:
                    message = self._queue.get()
                if isinstance(message, Exception):
                    raise message
        return message

    def close(self):
        if self._is_host is None:
            return

        if self._conn is not None:
            if self._is_host:
                # Send the close message
                if not self._conn.closed:
                    try:
                        _tcp_pickle_send(self._conn, {"message": "close"}, **self._transport_options)
                    except (EOFError, ConnectionError):
                        pass
            self._conn.close()
            self._conn = None

        # Wait for the worker thread to finish
        if self._receiver_thread is not None:
            if self._receiver_thread.is_alive():
                self._receiver_thread.join()
            self._receiver_thread = None

        if self._listener is not None:
            self._listener.close()
            self._listener = None


class TCPTransportClient:
    def __init__(self, **kwargs):
        self._protocol = TCPPickleProtocol(**kwargs)

    def get_run_server_function(self):
        kwargs = self._protocol.get_worker_configuration()
        return run_shm_server, kwargs

    def send(self, message):
        if self._protocol is None:
            raise ConnectionError("There is no active connection.")
        self._protocol.send(message)
        return self._protocol.receive()

    @_disable_reenter
    def send_interrupt(self, message):
        if self._protocol is None:
            raise ConnectionError("There is no active connection.")
        self._protocol.send(message, interrupt=True)

    def __enter__(self):
        self._protocol.start_host()
        return self

    def wait_for_connection(self, *args, **kwargs):
        del args, kwargs
        self._protocol.wait_for_worker()

    def close(self):
        if self._protocol is not None:
            self._protocol.close()
            self._protocol = None
        
    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.close()


def run_shm_server(*, 
                   handle,
                   handle_interrupt,
                   **kwargs):
    interrupt_result_queue = Queue()
    def worker_interrupt(protocol, handle_interrupt, interrupt_result_queue):
        try:
            while True:
                msg = protocol.receive(interrupt=True)
                handle_interrupt(msg)
        except BaseException as e:
            interrupt_result_queue.put(e)
            return

    safe_terminate = False
    try:
        protocol = TCPPickleProtocol(**kwargs)
        protocol.connect_worker()
        interrupt_thread = threading.Thread(
            target=worker_interrupt, 
            args=(protocol, handle_interrupt, interrupt_result_queue))
        interrupt_thread.start()
        logging.info(f"Connection accepted, protocol: {protocol.protocol_name}")
        while interrupt_thread.is_alive():
            msg = protocol.receive()
            if msg.get("message") == "close":
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
    finally:
        protocol.close()
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
                cancel_token.cancel_if_requested()
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
