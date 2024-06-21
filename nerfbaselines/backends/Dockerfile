FROM ubuntu:22.04

RUN apt-get update --fix-missing && \
    apt-get install -y wget \
                       bzip2 \
                       build-essential \
                       ca-certificates \
                       curl \
                       libgtk2.0-dev \
                       ffmpeg libsm6 libxext6 \
                       git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1

RUN echo "[global]\nno-cache-dir = True\n[install]\ncompile = no\n" > /etc/pip.conf && \
    curl -L -o ~/conda.sh -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash ~/conda.sh -b -u -p /opt/conda && \
    rm -rf ~/conda.sh && \
    /opt/conda/bin/conda clean -ay && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH

# Fix permissions for singularity - not needed for docker
RUN chmod -R go=u $HOME && \
    chmod -R go=u /opt/conda
