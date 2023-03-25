#!/bin/bash
# Immediately exit on any errors.
set -e
# Print commands as they are run.
set -o xtrace

docker run \
    --tty \
    --interactive \
    --network host \
    --dns=1.1.1.1 \
    --privileged \
    --gpus all \
    --device /dev/dri:/dev/dri \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    ihmcrobotics/nvidia-pytorch-train:0.4 bash