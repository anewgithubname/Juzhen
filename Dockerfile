FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

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
    clang-15 

RUN apt-get install -y libopenblas-dev cmake unzip libboost-dev

# Install CUDA toolkit
RUN echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

RUN nvcc --version
RUN git clone https://github.com/anewgithubname/Juzhen.git
WORKDIR /Juzhen
RUN unzip dataset.zip
RUN mkdir build 
WORKDIR /Juzhen/build
RUN cmake .. -DCMAKE_CXX_COMPILER=clang++-15
RUN cmake --build . --config Release -j 16

WORKDIR /Juzhen