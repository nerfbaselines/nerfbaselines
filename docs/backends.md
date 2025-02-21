# Isolated backends

## Introduction
In order to run the different methods, each of which uses different Python version and a set of dependencies (Python/C++), NerfBaselines relies on the concept of backends. A backend is an encapsulated container that contains the necessary dependencies to run a specific method.
The main package from which you can access these backends will be called **host**. The host orchestrates the backends and provides a unified interface for running the methods. Currently, there are three types of backends:
- **Conda**: The Conda backend runs the method in a Conda environment (with both Python and C++ dependencies).
- **Docker**: The Docker backend pulls (from the NerfBaselines registry) and runs the method in a Docker container. For some methods, this container may not be available, in which case the backend will build the container from the Dockerfile.
- **Apptainer**: The Apptainer backend runs the method in a Singularity container. When Docker image is available, it pulls and runs it, otherwise, it pulls a generic NerfBaselines image and builds the method inside the container as Conda backend.

When unsure, we recommend starting with the **Conda** backend as it is the easiest to setup. It may not provide sufficient isolation and may fail on your system. In that case you can choose either **Docker** or **Apptainer** backend - the latter is recommended for HPC systems.
Also note, that the **Docker** backend is the fastest to install in most cases at we release pre-built images for most methods.

The first time you run a method, the backend will download the necessary dependencies and build the method. This may take some time, but subsequent runs will be faster. After the backend is ready, NerfBaselines starts it in a separate process and connects to it using the implemented communication protocols.

## Communication with the backend
NerfBaselines uses a custom communication protocol to interface with the backend (**worker**) process.
The communication protocol is used to send and receive messages between the **host** and **worker** processes. The messages are serialized using the Pickle protocol. 
The highest version of the protocol supported by both ends is chosen and if it is supported, pickle will use separate buffers for binary data (numpy arrays). The communication protocol will attempt to setup shared memory buffers to transfer the buffers between the **host** and **worker** processes faster.
There are currently two low-level protocols implemented: 1) UNIX pipes, 2) TCP sockets. The UNIX pipes are used by default, but if they are not available (or manually disabled), the communication will fallback to using TCP sockets. Note, that TCP sockets could be slower on some HPC systems and pipes are recommended.

After the connection is established, you will see the established protocol string in the logs. The format is the folloging:
```
{protocol}-pickle{pickle-version}-shm{shm-size}
```
where `{protocol}` is either `pipe` or `tcp`, `{pickle-version}` is the used pickle protocol and should be 5 to achieve best performance. `{shm-size}` is the size of the shared memory buffer used to transfer the data between the processes, and has the format `{size}{unit}`, where the unit can be empty or `K`, `M`, or `G`. The protocol can chosen with the folloging environment variables:
- `NERFBASELINES_PROTOCOL={protocol}[-pickle{version}][-shm{size}{unit}]`, where `{protocol}` can also be `auto` and the rest of the string is optional, but otherwise the format is the same as in the logs.
- `NERFBASELINES_TCP_PORT={port}` - TCP port (e.g., `12345`), default is `0` (random port)
- `NERFBASELINES_TCP_HOSTNAME={host}` - TCP host name (e.g. `localhost`)
- `NERFBASELINES_SHM_SIZE={size}` - the size of the shared memory buffer (in bytes)


```{note}
The default value of NERFBASELINES_PROTOCOL is `auto-pickle5-72M`. This means the communication will start using UNIX pipes and will fallback to using TCP if the pipes are not available. It will then attempt to use pickle5 protocol and fallback to highest supported pickle protocol on both ends. Finally, it will try to setup a shared memory buffer of 72MB size, and fallback to not using shared memory if it fails.
If Python 3.8+ and UNIX is used on both ends, the best performance should be achieved.
```

## Zero-copy communication
By default, the communication protocol tries to be safe and does not assume anything about how it is used. However, this means
for each message, two memory copies are made - one to move tensors to the shared memory buffer and one to move them to the receiving process. Sometimes, this is not necessary and the user can opt-in to zero-copy communication.

### Avoid copy on caller site
When recieving a message, the memory is copied from the shared memory buffer to the receiving process. This is done in order to release the shared memory so it can be used by new messages without overriding the old data. If the recieved data are used **within the lifetime of the message**, the user can avoid this memory copy and instead use the shared memory buffer directly. In NerfBaselines this is internally used for faster trajectory rendering and faster viewer.

```{warning}
The user should be extremely careful when using this feature as the shared memory buffer is reused for new messages. This means that the data will be overwritten when a new message is sent. After recieving the message, the user should fully process and discard the data before sending a new message.
```

To enable zero-copy, the user can place code inside `nerfbaselines.backends.zero_copy` context manager. This will cause all backend calls within the context to use zero-copy communication.

```python
from nerfbaselines.backends import zero_copy

with zero_copy():
    # All backend calls within this context will use zero-copy communication
    result = backend.render(...)

    # Save the result to disk
    ...

    del result

    result2 = backend.render(...)
    # WARNING: at this point, all tensors in result are invalid and should not be used
    ...
```

### Avoiding copy on worker site
The NerfBaselines interface expects each `render` method to return a dictionary of `np.ndarrays`.
However, most methods process data in CUDA memory and returning a `np.ndarray` requires moving the data to the host memory.
The communication protocol then requires another memory copy to move the data to the shared memory buffer.
Most currently implemented methods already avoid this additional memory copy by allocating
`np.ndarray` directly in shared memory and copying the data from CUDA directly to the shared buffer.
This can also be implemented for your own method. 
In the following example, we use the `nerfbaselines.backend.backend_allocate_ndarray` function:
```python
# We start with a CUDA tensor
cuda_tensor: torch.Tensor = ... 

# We allocate a new shared memory buffer
dtype_name = str(cuda_tensor.dtype).split('.')[-1]
np_array = backend_allocate_ndarray(cuda_tensor.shape, dtype=dtype_name)

# Now, we copy the data from the CUDA tensor to the shared memory buffer
torch.from_numpy(np_array).copy_(cuda_tensor)
```
