# Docker image for training Pytorch models for computer vision tasks.
# Current version: 0.1

FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install commonly required libraries and packages
# Note: Removing /var/lib/apt/lists/* at the end can reduce the
# compressed image size by about 25 MB.
RUN apt-get --yes update

RUN apt-get --yes install \
    nano \
    git \
    wget \
    curl \
    unzip \
    apt-transport-https \
    iputils-ping \
    ca-certificates \
    curl \
    software-properties-common \
    python3-opencv \
    sudo \
    iproute2 \
    python3-pip \
    > /dev/null \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade gluoncv

# Setup a robotlab user as the development user, to avoid using root.
RUN addgroup robotlab \
 && adduser --home /home/robotlab --gecos "Rosie Robot,1117,1234567,2345678" --ingroup robotlab --disabled-password robotlab \
 && chown -R robotlab /home/robotlab \
 && adduser robotlab sudo \
 && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER robotlab
WORKDIR /home/robotlab

# Make sure the .config folder exists and is owned by the robotlab user.
# This is useful for later installed apps.
RUN mkdir -p /home/robotlab/.config
RUN chown -R robotlab:robotlab /home/robotlab/.config
