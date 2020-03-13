FROM nvidia/cuda:10.1-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y curl

RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
RUN chmod +x Anaconda3-2019.10-Linux-x86_64.sh
RUN ./Anaconda3-2019.10-Linux-x86_64.sh -b -p ~/anaconda3

WORKDIR /app

