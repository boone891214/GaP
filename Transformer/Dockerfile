# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.06-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace

RUN mkdir -p /root/.config/pip/
RUN echo '[global]' > /root/.config/pip/pip.conf
RUN echo 'index-url = https://mirrors.aliyun.com/pypi/simple/' >> /root/.config/pip/pip.conf

# RUN sed -i 's/^/#&/g' /etc/apt/sources.list.d/*
RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list
RUN sed -i "s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list
RUN sed -i "s/download.docker.com/mirrors.aliyun.com\/docker-ce/g" /etc/apt/sources.list
RUN sed -i "s/http:/https:/g" /etc/apt/sources.list
RUN echo 'Acquire::http::proxy "http://11.162.93.61:3128/";' > /etc/apt/apt.conf.d/99proxy
RUN echo 'Acquire::https::proxy "http://11.162.93.61:3128/";' >> /etc/apt/apt.conf.d/99proxy
RUN echo 'Acquire::ftp::proxy "http://11.162.93.61:3128/";' >> /etc/apt/apt.conf.d/99proxy

RUN git config --global http.proxy "http://11.162.93.61:3128/"
RUN git config --global https.proxy "http://11.162.93.61:3128/"
#RUN git clone https://github.com/NVIDIA/apex \
# && cd apex \
# && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# Install Python dependencies
RUN pip install --upgrade --no-cache-dir pip \
 && pip install --no-cache-dir \
      sacrebleu \
      sentencepiece
RUN pip install tensorboardX -i https://mirrors.aliyun.com/pypi/simple/

RUN apt-get update
RUN apt-get install -y cmake pkg-config protobuf-compiler libprotobuf-dev libgoogle-perftools-dev
RUN git clone https://github.com/google/sentencepiece.git /workspace/sentencepiece
RUN cd /workspace/sentencepiece \
  && git checkout d4dd947 \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j 8 \
  && make install \
  && ldconfig -v

ENV PYTHONPATH=/workspace/translation/examples/translation/subword-nmt/
WORKDIR /workspace/translation
RUN git clone https://github.com/rsennrich/subword-nmt.git /workspace/translation/examples/translation/subword-nmt/
RUN git clone https://github.com/NVIDIA/cutlass.git && cd cutlass && git checkout ed2ed4d6 && cd ..
COPY . .
RUN pip install -e .
RUN git clone https://github.com/NVIDIA/dllogger.git
RUN pip install dllogger/
