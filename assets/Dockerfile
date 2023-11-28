FROM "nvidia/cuda:11.8.0-devel-ubuntu22.04"

RUN apt-get update --fix-missing && \
    apt-get install -y wget \
                       bzip2 \
                       ca-certificates \
                       curl \
                       libgtk2.0-dev \
                       ffmpeg libsm6 libxext6 \
                       git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py311_23.9.0-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -u -p /opt/conda && \
    rm -rf ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ay && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH

# Fix permissions for singularity - not needed for docker
RUN chmod -R go=u $HOME && \
    chmod -R go=u /opt/conda
