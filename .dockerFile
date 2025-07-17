FROM ubuntu:24.04

# 비대화형 apt
ENV DEBIAN_FRONTEND=noninteractive \
    CC=gcc \
    CXX=g++

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      nvidia-cuda-toolkit \
      build-essential \
      gnuplot \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENTRYPOINT ["bash"]