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
NerfBaselines uses multiple communication protocols to interface with the backend (**worker**) process:
- **tcp-pickle**: The fallback communication protocol that works in most cases. It starts a TCP server in the **host** process and the **worker** process attach to it. The messages are serialized using the Pickle protocol. The highest version of the protocol supported by both ends is chosen and if it is supported, pickle will use separate buffers for binary data (numpy arrays). These are sent as separate messages.
- **shm-pickle**: This is the default communication protocol. It uses shared memory to transfer the data between the **host** and **worker** processes. The messages are serialized using the Pickle protocol. The highest version of the protocol supported by both ends is chosen and if it is supported, pickle will use separate buffers for binary data (numpy arrays). These are sent as separate messages. The current implementation uses spinlocks to synchronize access to the shared memory, which may be suboptimal on some systems.

The communication protocol can be selected by setting the environment variable `NERFBASELINES_PROTOCOL`. It accepts a comma-separated list of protocols, with the first one being the fallback one. After establishing the connection using the fallback protocol, the **host** and **worker** will initiate a protocol upgrade sequence where they will attempt the highest supported protocol (by both ends) from the list. It will take the list of protocols in the reverse order until a common protocol is found. If no common protocol is found, the connection will continue using the fallback protocol.

```{note}
The default value of NERFBASELINES_PROTOCOL is `tcp-pickle,shm-pickle`. This means the communication will start using the `tcp-pickle` protocol and will attempt to upgrade to `shm-pickle` if both ends support it. Note, shared memory is only supported if both processes are running on Linux/MacOS and both Python versions are 3.8+.
```
