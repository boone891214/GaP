

DOCKER_IMAGE=${1:-'transformer'}
DATA_DIR=${2:-'.'}
EXTRA=${3:-" "}

nvidia-docker run -it --rm --ipc=host -v ${DATA_DIR}/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v ${PWD}:/home -w /home -v ${PWD}/results:/results ${EXTRA} ${DOCKER_IMAGE}
