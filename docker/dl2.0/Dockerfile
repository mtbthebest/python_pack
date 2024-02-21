FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
RUN apt -y update
RUN apt-get install -y apt-transport-https

RUN  apt install -y \
    python3-pip \
    sudo \
    wget \
    git \
    vim
RUN apt  install -y \
    curl \
    ca-certificates \
    bzip2 \
    libx11-6 \
    build-essential \
    screen \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    graphviz \
    libopenmpi-dev \
    tmux \
    liblzma-dev \
    libbz2-dev


ARG USER=ubuntu
ARG PASSWD=ubuntu
ARG GROUP=admin
RUN useradd -m $USER && \
    echo "$USER:$PASSWD" | chpasswd && \
    echo "$USER ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "$USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install python
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt -y install wget build-essential libreadline-gplv2-dev libncursesw5-dev \
     libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz 
RUN tar xzf Python-3.10.0.tgz
WORKDIR /Python-3.10.0 
RUN ./configure --enable-optimizations  && make altinstall

WORKDIR /
RUN /usr/local/bin/python3.10 -m pip install --upgrade pip


RUN /usr/local/bin/python3.10 -m venv /env_dl
COPY ./config/env_req.txt /home/$USER
ENV os="ubuntu1804"
ENV tag="8.6.0-cuda-11.8"

RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.0/local_repos/nv-tensorrt-local-repo-ubuntu1804-8.6.0-cuda-11.8_1.0-1_amd64.deb 
RUN dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb && cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/ \
    &&  apt-get update && apt-get -y install tensorrt
RUN . /env_dl/bin/activate && pip install -r /home/$USER/env_req.txt

USER root

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /run/sshd
RUN mkdir /home/$USER/.ssh/
RUN echo 'ubuntu:rootpassws' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
COPY ./config/dl2_4.pub /home/$USER/.ssh/authorized_keys

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN chown -R $USER:$USER /env_dl/*
WORKDIR /