# syntax=docker/dockerfile:1
FROM ubuntu:latest
WORKDIR /app
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update && apt-get -y install python3-pip
RUN pip3 install numpy matplotlib casadi scipy ipython
RUN add-apt-repository ppa:fenics-packages/fenics
RUN apt update
RUN apt -y install fenicsx

ENV PYTHONPATH "/:/app/"
COPY entrypoint.sh /
COPY . .
ENTRYPOINT ["/entrypoint.sh"]
