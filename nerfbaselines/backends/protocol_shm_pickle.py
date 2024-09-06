import sys
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


def _shm_recv(shared_memory):
    num_buffers, = struct.unpack("I", shared_memory.buf[_SHM_OFFSET:_SHM_OFFSET+4])
    len_header = 4+8*num_buffers
    buff_lens = struct.unpack("Q"*num_buffers, shared_memory.buf[_SHM_OFFSET+4:_SHM_OFFSET+len_header])
    write_first = buff_lens[0] + _SHM_OFFSET+len_header <= shared_memory.size
    if not write_first:
        _shm_set_flag(shared_memory, 5)
    buffers = []
    for buff_i, buff_len in enumerate(buff_lens):
        # Write first
        if buff_i == 0 and write_first:
            # Copy bytes from the shared_memory.buf
            buffers.append(shared_memory.buf[_SHM_OFFSET+len_header:_SHM_OFFSET+len_header+buff_len].tobytes())
            _shm_set_flag(shared_memory, 5)
            continue

        # Allocate buffer
        buffer = bytearray(buff_len)
        buffers.append(buffer)
        
        for i in range(0, buff_len, shared_memory.size - _SHM_OFFSET):
            mess_len = min(buff_len - i, shared_memory.size - _SHM_OFFSET)
            _shm_wait_for(shared_memory, [4])
            memoryview(buffer)[i:i+mess_len] = shared_memory.buf[_SHM_OFFSET:_SHM_OFFSET+mess_len]
            _shm_set_flag(shared_memory, 5)
    return pickle.loads(buffers[0], **({'buffers': buffers[1:]} if len(buffers) > 1 else {}))  # type: ignore


def _shm_send(shared_memory, message, set_flag,
          *,
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
    header = struct.pack("I", len(buffers))
    header += struct.pack("Q"*len(buffers), *(len(buff) for buff in buffers))
    shared_memory.buf[_SHM_OFFSET:_SHM_OFFSET+len(header)] = header
    write_first = len(buffers[0]) + _SHM_OFFSET+len(header) <= shared_memory.size
    if write_first:
        shared_memory.buf[_SHM_OFFSET+len(header):_SHM_OFFSET+len(header)+len(buffers[0])] = buffers[0]
        buffers = buffers[1:]
    _shm_set_flag(shared_memory, set_flag)
    _shm_wait_for(shared_memory, [5])
    for buffer in buffers:
        for i in range(0, len(buffer), shared_memory.size - _SHM_OFFSET):
            mess_len = min(len(buffer) - i, shared_memory.size - _SHM_OFFSET)
            shared_memory.buf[_SHM_OFFSET:_SHM_OFFSET+mess_len] = buffer[i:i+mess_len]
            _shm_set_flag(shared_memory, 4)
            _shm_wait_for(shared_memory, [5])



def _shm_recv_unify_buffers(shared_memory):
    # NOTE: This function is not used in the current implementation
    # There are no performance benefits for standard workloads
    # And it currently fails for large messages
    num_buffers, = struct.unpack("I", shared_memory.buf[_SHM_OFFSET:_SHM_OFFSET+4])
    buff_lens = struct.unpack("Q"*num_buffers, shared_memory.buf[_SHM_OFFSET+4:_SHM_OFFSET+4+8*num_buffers])
    offset = 4+8*num_buffers
    total_len = sum(buff_lens)
    data = bytearray(total_len)
    sh_offset = offset

    for i in range(offset, total_len+offset, shared_memory.size - _SHM_OFFSET):
        mess_len = min(total_len+offset - i, shared_memory.size - _SHM_OFFSET)
        if i > offset:
            _shm_wait_for(shared_memory, [4])
        start, end = _SHM_OFFSET+sh_offset, min(_SHM_OFFSET+sh_offset+mess_len, shared_memory.size)
        print(f"read {start}:{end}, len({end-start}), lenwanted({mess_len})")
        start, end = i-offset, min(i-offset+mess_len, total_len)
        print(f"write {start}:{end}, len({end-start}), lenwanted({mess_len})")
        memoryview(data)[i-offset:i-offset+mess_len] = shared_memory.buf[_SHM_OFFSET+sh_offset:_SHM_OFFSET+sh_offset+mess_len]
        sh_offset = 0
        _shm_set_flag(shared_memory, 5)

    offset = 0
    buffers = []
    for buff_len in buff_lens:
        buffer = memoryview(data)[offset:offset+buff_len]
        buffers.append(buffer)
        offset += buff_len
    out = pickle.loads(buffers[-1], **({'buffers': buffers[:-1]} if len(buffers) > 1 else {}))  # type: ignore
    return out


def _shm_send_unify_buffers(shared_memory, message, set_flag,
          *,
          pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
          pickle_use_buffers: bool = False):
    # NOTE: This function is not used in the current implementation
    # There are no performance benefits for standard workloads
    # And it currently fails for large messages
    buffers = []
    def _add_buffer(buffer):
        buffers.append(buffer.raw())
    buffers.append(
        pickle.dumps(message, 
                     protocol=pickle_protocol,
                     **({ "buffer_callback": _add_buffer } 
                        if (pickle_use_buffers and pickle_protocol >= 5) 
                        else {})))  # type: ignore
    header = struct.pack("I", len(buffers))
    header += struct.pack("Q"*len(buffers), *(len(buff) for buff in buffers))
    shared_memory.buf[_SHM_OFFSET:_SHM_OFFSET+len(header)] = header
    offset = len(header)

    for buffer in buffers:
        local_offset = 0
        rest = min(shared_memory.size-_SHM_OFFSET - offset, len(buffer) - local_offset)
        shared_memory.buf[_SHM_OFFSET+offset:_SHM_OFFSET+offset+rest] = buffer[local_offset:local_offset+rest]
        local_offset += rest
        offset += rest
        while local_offset < len(buffer):
            # Flush buffer
            _shm_set_flag(shared_memory, set_flag)
            set_flag = 4
            _shm_wait_for(shared_memory, [5])
            offset = 0

            rest = min(shared_memory.size-_SHM_OFFSET - offset, len(buffer) - local_offset)
            shared_memory.buf[_SHM_OFFSET+offset:_SHM_OFFSET+offset+rest] = buffer[local_offset:local_offset+rest]
            local_offset += rest
            offset += rest

    # Flush last data
    _shm_set_flag(shared_memory, set_flag)
    set_flag = 4
    _shm_wait_for(shared_memory, [5])


def _shm_wait_for(shared_memory, lock_value, sleep=0.00001, timeout=None):
    # print("waitf", lock_value, os.getpid())
    start_time = time.time()
    if shared_memory.buf is None:
        raise ConnectionClosed()
    flag = struct.unpack("i", shared_memory.buf[:4])[0]
    while flag not in lock_value and flag != 7:
        time.sleep(sleep)
        if shared_memory.buf is None:
            raise ConnectionClosed()
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError()
        flag = struct.unpack("i", shared_memory.buf[:4])[0]
    if flag == 7:
        raise ConnectionClosed()
    return flag


def _shm_set_flag(shared_memory, lock_value):
    # print("  setf", lock_value, os.getpid())
    shared_memory.buf[:4] = struct.pack("i", lock_value)


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
        _shm_set_flag(self._shared_memory, 0)

    def wait_for_worker(self, timeout=None):
        assert self._is_host is not None, "Not started as host or worker"

        # Establish the protocol
        _shm_wait_for(self._shared_memory, [3], timeout=timeout)
        msg = _shm_recv(self._shared_memory)
        assert msg["message"] == "ready", f"Unexpected message {msg['message']}"
        self._transport_options = {}
        transport_options = msg["transport_options"]
        transport_options["pickle_protocol"] = min(
            transport_options["pickle_protocol"],
            pickle.HIGHEST_PROTOCOL)
        transport_options["pickle_use_buffers"] = (
            transport_options.get("pickle_use_buffers", False) and 
            transport_options["pickle_protocol"] >= 5)
        _shm_wait_for(self._shared_memory, [6])
        _shm_send(self._shared_memory, {
            "message": "ready_ack",
            "transport_options": transport_options,
        }, 1, pickle_protocol=1, pickle_use_buffers=False)
        self._transport_options = transport_options
        self._has_server = True

        # Start the communication by releasing the channel
        self._connected = True
        _shm_set_flag(self._shared_memory, 0)

    def get_worker_configuration(self):
        assert self._is_host is True, "Not started as host"
        assert self._shared_memory is not None, "Not initialized"
        return {
            "shared_memory_name": self._shared_memory.name,
        }

    def connect_worker(self):
        self._is_host = False
        self._attach_worker_resources()

        # Establish the protocol
        # pipe_out = os.open(path + "/pipe-out", os.O_RDONLY)
        # pipe_out_interrupt = os.open(path + "/pipe-out-interrupt", os.O_RDONLY)
        # pipe_in = os.open(path + "/pipe-in", os.O_WRONLY)

        _shm_wait_for(self._shared_memory, [0])
        _shm_send(self._shared_memory, { 
            "message": "ready",
            "transport_options": {
                "pickle_protocol": pickle.HIGHEST_PROTOCOL,
                "pickle_use_buffers": pickle.HIGHEST_PROTOCOL >= 5,
            }
        }, 3, **self._transport_options)
        _shm_set_flag(self._shared_memory, 6)
        _shm_wait_for(self._shared_memory, [1])
        setup_response = _shm_recv(self._shared_memory)
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
            _shm_wait_for(self._shared_memory, [0])
            _shm_send(self._shared_memory, message, channel, **self._transport_options)
            _shm_set_flag(self._shared_memory, 0)

    def receive(self, interrupt=False):
        assert self._is_host is not None, "Not started as host or worker"
        assert not self._is_host or not interrupt, "Only worker can receive interrupt messages"
        with self._protect_singlerun("receive", interrupt):
            channel = 3 if self._is_host else (2 if interrupt else 1)
            _shm_wait_for(self._shared_memory, [channel])
            return _shm_recv(self._shared_memory)

    def close(self):
        if self._is_host is None:
            return
        if self._shared_memory is not None:
            for _ in range(100):
                _shm_set_flag(self._shared_memory, 7)
            if self._is_host:
                self._shared_memory.unlink()
            self._shared_memory.close()
            self._shared_memory = None
