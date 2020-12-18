# FROM ubuntu:20.04
FROM nvidia/cudagl:11.0-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    vim \
    python3-pip \
    lsb-release \
    gnupg

WORKDIR requirements/
ADD requirements.txt /requirements
RUN pip3 install -r requirements.txt

# Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

WORKDIR /root/workspace
