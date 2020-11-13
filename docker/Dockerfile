FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV TZ=Europe/Moscow
ENV TERM xterm-256color

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y \
	python3-pip \
	python3-tk \
	libboost-all-dev

RUN apt-get -y install git
RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O speech_commands_v0.01.tar.gz
RUN mkdir speech_commands && tar -C speech_commands -xvzf speech_commands_v0.01.tar.gz 1> log
COPY mv.sh .
ENTRYPOINT mv.sh