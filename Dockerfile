FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt update && apt upgrade -y && \
    apt install -y git build-essential python3 python3-setuptools python3-dev python3-tk python3-pip swig python3-venv && \
    mkdir /app

WORKDIR /app
RUN python3 -m venv env && \
    source env/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache wheel 'conan<2.0.0' cmake parse pandas


WORKDIR /tmp
RUN source /app/env/bin/activate && \
    conan profile new default --detect && \
    conan profile update settings.compiler.libcxx=libstdc++11 default && \
    git clone https://github.com/AVSLab/basilisk.git --depth 1 --branch 2.1.6 --single-branch && \
    cd basilisk && \
    python conanfile.py --generator 'Unix Makefiles' --buildProject True
RUN source /app/env/bin/activate && \
    git clone https://github.com/connorfuhrman/paramz --depth 1 --branch connorfuhrman/np_type_alias_dep --single-branch && \
    pip install ./paramz

WORKDIR /app
COPY requirements.txt .
RUN source env/bin/activate && \
    pip install --no-cache -r requirements.txt
COPY attitude_pointing_rl.py app.py
