# IMAGENET_DIR="/home/admin/data/imagenet"
# PRETRAINED_DIR="/home/admin/nas/sparsity/pretrained/"
# sudo /maintenance/docker-gpu.sh -it -v  ${IMAGENET_DIR}:/data/imagenet -v $PWD:/workspace/rn50 -v ${PRETRAINED_DIR}:/data/pretrained/ --ipc=host nvidia_rn50 /maintenance/init_inner.sh /bin/bash

#sudo /maintenance/docker-gpu.sh -v xxxx -v xxxx /maintenance/init_inner.sh nvidia_rn50


PRETRAINED_DIR=/home/admin/nas/sparsity/pretrained/
DATA_DIR=/home/admin/nas/sparsity/data/Transformer/data/wmt14_en_de_joined_dict/
sudo /maintenance/docker-gpu.sh -it \
  -v $PWD/results:/results \
  -v $DATA_DIR:/data \
  -v $PWD:/home \
  -v ${PRETRAINED_DIR}:/data/pretrained/ \
  --ipc=host \
  --shm-size=1g \
  transformer_pyt /maintenance/init_inner.sh /bin/bash
