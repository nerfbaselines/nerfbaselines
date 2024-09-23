import threading
import os
import contextlib
import struct
import time
import pickle
from multiprocessing.shared_memory import SharedMemory
# import tempfile


_SHM_OFFSET = 8
_SHM_SIZE = 128 * 1024 * 1024  # 128 MB


class ConnectionClosed(ConnectionError):
    def __init__(self):
        super().__init__("Connection closed")


def _shm_recv(shared_memory, wait_for, timeout=None):
    buffers = []
    with _shm_wait_set_flag(shared_memory, wait_for, 5, timeout=timeout):
        num_buffers, shared_memory_size = struct.unpack("!IQ", shared_memory.buf[_SHM_OFFSET:_SHM_OFFSET+12])
        len_header = 12+8*num_buffers
        buff_lens = struct.unpack("!"+"Q"*num_buffers, shared_memory.buf[_SHM_OFFSET+12:_SHM_OFFSET+len_header])
        # print("recv", num_buffers, [buff for buff in buff_lens], "thread", threading.get_ident())
        write_first = buff_lens[0] + _SHM_OFFSET+len_header <= shared_memory_size
        if write_first:
            # Copy bytes from the shared_memory.buf
            buff_len = buff_lens[0]
            buffers.append(shared_memory.buf[_SHM_OFFSET+len_header:_SHM_OFFSET+len_header+buff_len].tobytes())
            buff_lens = buff_lens[1:]

    for buff_len in buff_lens:
        # Allocate buffer
        buffer = bytearray(buff_len)
        buffers.append(buffer)
        
        for i in range(0, buff_len, shared_memory_size - _SHM_OFFSET):
            mess_len = min(buff_len - i, shared_memory_size - _SHM_OFFSET)
            with _shm_wait_set_flag(shared_memory, [4], 5):
                memoryview(buffer)[i:i+mess_len] = shared_memory.buf[_SHM_OFFSET:_SHM_OFFSET+mess_len]
    return pickle.loads(buffers[0], **({'buffers': buffers[1:]} if len(buffers) > 1 else {}))  # type: ignore


def _shm_send(shared_memory, message, channel,
          *,
          wait_for,
          set_flag,
          shared_memory_size: int,
          pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
          pickle_use_buffers: bool = False):
    buffers = []
    def _add_buffer(buffer):
        buffers.append(buffer.raw())
    buffers.insert(0, 
        pickle.dumps(message, protocol=pickle_protocol,
                     **({ "buffer_callback": _add_buffer } 
                        if (pickle_use_buffers and pickle_protocol >= 5) 
                        else {})))  # type: ignore
    header = struct.pack("!IQ", len(buffers), shared_memory_size)
    header += struct.pack("!"+"Q"*len(buffers), *(len(buff) for buff in buffers))
    write_first = len(buffers[0]) + _SHM_OFFSET+len(header) <= shared_memory_size
    # print("send", len(buffers), [len(buff) for buff in buffers], "thread", threading.get_ident())
    with _shm_wait_set_flag(shared_memory, wait_for, channel):
        shared_memory.buf[_SHM_OFFSET:_SHM_OFFSET+len(header)] = header
        if write_first:
            shared_memory.buf[_SHM_OFFSET+len(header):_SHM_OFFSET+len(header)+len(buffers[0])] = buffers[0]
            buffers = buffers[1:]
    for buffer in buffers:
        for i in range(0, len(buffer), shared_memory_size - _SHM_OFFSET):
            mess_len = min(len(buffer) - i, shared_memory_size - _SHM_OFFSET)
            with _shm_wait_set_flag(shared_memory, [5], 4):
                shared_memory.buf[_SHM_OFFSET:_SHM_OFFSET+mess_len] = buffer[i:i+mess_len]
    with _shm_wait_set_flag(shared_memory, [5], set_flag):
        pass


@contextlib.contextmanager
def _shm_wait_set_flag(shared_memory, wait_for, lock_value, sleep=0.00001, timeout=None):
    lock_id = os.urandom(4)
    start_time = time.time()
    if shared_memory.buf is None:
        raise ConnectionClosed()
    # print("  waitf", lock_value, os.getpid(), threading.get_ident())
    # wp = False
    while True:
        if shared_memory.buf is None:
            raise ConnectionClosed()
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError()
        _data = shared_memory.buf[:8].tobytes()
        flag = struct.unpack("!I", _data[:4])[0]
        if flag == 7:
            raise ConnectionClosed()

        _lock_id = _data[4:]
        if _lock_id == lock_id and flag == 16:
            # Lock was acquired
            yield

            # Release the lock
            shared_memory.buf[:8] = struct.pack("!I", lock_value) + b'\x00' * 4
            break

        # If flag is in wait_for, we try to acquire the lock
        if flag in wait_for:
            _data = struct.pack("!I", 16) + lock_id
            shared_memory.buf[:8] = _data
        else:
            time.sleep(sleep)


def _remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """
    from multiprocessing import resource_tracker

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:  # type: ignore
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]  # type: ignore


class SharedMemoryProtocol:
    def __init__(self,
                 *,
                 shared_memory_name=None,
                 shared_memory_size=int(os.environ.get("NERFBASELINES_SHARED_MEMORY_SIZE", _SHM_SIZE))):
        self._shared_memory_name = shared_memory_name
        self._shared_memory = None
        self._is_host = None
        self._connected = False
        self._transport_options = {
            "pickle_protocol": 1,
            "pickle_use_buffers": False,
            "shared_memory_size": shared_memory_size,
        }
        self._singlerun_contexts = {}
        self._shared_memory_size = shared_memory_size

        self._attach_worker_resources()

    def _attach_worker_resources(self):
        # Test if shared memory is available to fail early
        if self._shared_memory_name is not None and self._shared_memory is None:
            # Fix a bug in multiprocessing.resource_tracker
            _remove_shm_from_resource_tracker()
            self._shared_memory = SharedMemory(name=self._shared_memory_name, create=False)

    def start_host(self):
        self._is_host = True
        
        # Create the shared memory
        self._shared_memory = SharedMemory(name=self._shared_memory_name,
                                           size=self._shared_memory_size, 
                                           create=True)
        self._shared_memory.buf[:8] = b"\x00" * 8

    def wait_for_worker(self, timeout=None):
        assert self._is_host is not None, "Not started as host or worker"

        # Establish the protocol
        msg = _shm_recv(self._shared_memory, wait_for=[3], timeout=timeout)
        assert msg["message"] == "ready", f"Unexpected message {msg['message']}"
        transport_options = msg["transport_options"]
        transport_options["pickle_protocol"] = min(
            transport_options["pickle_protocol"],
            pickle.HIGHEST_PROTOCOL)
        transport_options["pickle_use_buffers"] = (
            transport_options.get("pickle_use_buffers", False) and 
            transport_options["pickle_protocol"] >= 5)
        old_transport_options = self._transport_options
        self._transport_options = transport_options
        self._has_server = True
        self._connected = True
        _shm_send(self._shared_memory, {
            "message": "ready_ack",
            "transport_options": transport_options,
        }, 1, wait_for=[6], set_flag=0, **old_transport_options)

    def get_worker_configuration(self):
        assert self._is_host is True, "Not started as host"
        assert self._shared_memory is not None, "Not initialized"
        return {
            "shared_memory_name": self._shared_memory.name,
            "shared_memory_size": self._shared_memory.size,
        }

    def connect_worker(self):
        self._is_host = False
        self._attach_worker_resources()

        # Establish the protocol
        # pipe_out = os.open(path + "/pipe-out", os.O_RDONLY)
        # pipe_out_interrupt = os.open(path + "/pipe-out-interrupt", os.O_RDONLY)
        # pipe_in = os.open(path + "/pipe-in", os.O_WRONLY)

        _shm_send(self._shared_memory, { 
            "message": "ready",
            "transport_options": {
                "pickle_protocol": pickle.HIGHEST_PROTOCOL,
                "pickle_use_buffers": pickle.HIGHEST_PROTOCOL >= 5,
                "shared_memory_size": self._shared_memory_size,
            },
        }, 3, wait_for=[0], set_flag=6, **self._transport_options)
        setup_response = _shm_recv(self._shared_memory, wait_for=[1])
        if setup_response["message"] != "ready_ack":
            raise RuntimeError(f"Unexpected message {setup_response['message']}")
        self._connected = True
        self._transport_options = setup_response["transport_options"]

    @contextlib.contextmanager
    def _protect_singlerun(self, *args):
        if args in self._singlerun_contexts:
            raise RuntimeError("Re-entering the same function is not allowed.")
        self._singlerun_contexts[args] = True
        try:
            yield
        finally:
            del self._singlerun_contexts[args]

    @property
    def protocol_name(self):
        protocol_name = f"shm-pickle{self._transport_options['pickle_protocol']}"
        if self._transport_options["pickle_use_buffers"]:
            protocol_name += "-buffers"
        return protocol_name

    def send(self, message, interrupt=False):
        assert self._is_host is not None, "Not started as host or worker"
        assert self._is_host or not interrupt, "Only host can send interrupt messages"
        with self._protect_singlerun("send", interrupt):
            channel = 3 if not self._is_host else (2 if interrupt else 1)
            _shm_send(self._shared_memory, message, channel, wait_for=[0], set_flag=0, **self._transport_options)

    def receive(self, interrupt=False):
        assert self._is_host is not None, "Not started as host or worker"
        assert not self._is_host or not interrupt, "Only worker can receive interrupt messages"
        with self._protect_singlerun("receive", interrupt):
            channel = 3 if self._is_host else (2 if interrupt else 1)
            return _shm_recv(self._shared_memory, wait_for=[channel])

    def close(self):
        if self._is_host is None:
            return
        if self._shared_memory is not None:
            for _ in range(100):
                self._shared_memory.buf[:8] = struct.pack("!II", 7, 0)
            if self._is_host:
                self._shared_memory.unlink()
            self._shared_memory.close()
            self._shared_memory = None
