# FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
FROM ubuntu:latest

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install necessary packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    vim \
    cmake \
    git \
    gdb \
    clang-12 

RUN apt-get install -y libopenblas-dev cmake unzip libboost-dev

# Install CUDA toolkit
# RUN echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && \
    # echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# RUN nvcc --version
