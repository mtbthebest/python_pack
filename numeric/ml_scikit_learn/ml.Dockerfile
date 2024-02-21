FROM nvidia/cuda:11.8.0-devel-ubuntu18.04
FROM python:3.9.10
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
    graphviz \
    tmux


ARG USER=ubuntu
ARG PASSWD=ubuntu
ARG GROUP=admin
RUN useradd -m $USER && \
    echo "$USER:$PASSWD" | chpasswd && \
    echo "$USER ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "$USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN python -m pip install --upgrade pip



RUN python -m venv /env_ml
COPY ./ml_env_req.txt /
RUN . /env_ml/bin/activate && pip install -r /ml_env_req.txt

USER root

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /run/sshd
RUN mkdir /home/$USER/.ssh/
RUN echo 'ubuntu:rootpassws' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
COPY ./container.pub /home/$USER/.ssh/authorized_keys

ENTRYPOINT service ssh restart &&  bash

WORKDIR /


