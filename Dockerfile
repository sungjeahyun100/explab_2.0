FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gnuplot \
      libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENTRYPOINT ["bash"]