# VISIBLE_DEVICE=${1:-'0,1,2,3,4,5,6,7'}
PRETRAINED_DIR=/proj/usr/haoran.li/sparsity/pretrained/
DATA_DIR=/proj/usr/haoran.li/sparsity/data/Transformer/data/wmt14_en_de_joined_dict/
docker run -it --rm \
  --gpus all \
  --ipc=host \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $PWD/results:/results \
  -v $DATA_DIR:/data \
  -v $PWD:/home \
  -v ${PRETRAINED_DIR}:/data/pretrained/ \
  transformer_pyt
