#!/bin/bash

docker run -it --rm \
  --gpus all \
  --ipc=host \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $PWD/results:/results \
  -v $PWD/data:/data \
  -v $PWD:/home \
  transformer_pyt
