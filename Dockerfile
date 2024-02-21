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
    libopenmpi-dev


ARG USER=ubuntu
ARG PASSWD=ubuntu
ARG GROUP=admin
RUN useradd -m $USER && \
    echo "$USER:$PASSWD" | chpasswd && \
    echo "$USER ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "$USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN python -m pip install --upgrade pip



RUN python -m venv /env_nlp
COPY ./config/env_nlp_req.txt /home/$USER
RUN . /env_nlp/bin/activate && pip install -r /home/$USER/env_nlp_req.txt

# RUN chown -R $USER:$USER /home/$USER/volume/env_nlp

USER root

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /run/sshd
RUN mkdir /home/$USER/.ssh/
RUN echo 'ubuntu:rootpassws' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
COPY ./config/container.pub /home/$USER/.ssh/authorized_keys

RUN apt install tmux

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# ENTRYPOINT sudo service ssh restart &&  bash
ENTRYPOINT service ssh restart &&  bash
# # ENTRYPOINT chown -R $USER:$USER /home/ubuntu/volume/env_nlp  && bash 
#&& . /home/$USER/volume/env_nlp/bin/activate

# USER $USER
# CMD  . /env_nlp/bin/activate && jupyter-lab --port 9000 --allow-root --ip=0.0.0.0 --LabApp.token='' --no-browser
# CMD . /env_nlp/bin/activate && jupyter-lab --port 9000 --allow-root --ip=0.0.0.0 --LabApp.token='' --no-browser 
WORKDIR /
# ENV PATH=/env_nlp/bin/bin:$PATH


#
#jupyter-lab --port 8888 --allow-root --ip=0.0.0.0 --LabApp.token='' --no-browser
#jupyter-lab --port 9000  --allow-root --ip=0.0.0.0 --LabApp.token='' --no-browser
#
# tmux new-session -d -s rtb123 && \
#                 tmux send-keys '. /env_nlp/bin/activate && jupyter-lab --port 9000 --allow-root --ip=0.0.0.0 --LabApp.token="" --no-browser' C-m && \
#                 tmux detach -s rtb123 &&   bash