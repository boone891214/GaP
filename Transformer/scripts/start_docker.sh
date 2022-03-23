

DOCKER_IMAGE=${1:-'transformer'}
EXTRA=${2:-" "}

nvidia-docker run -it --rm --ipc=host -v ${PWD}/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v ${PWD}:/home -w /home -v ${PWD}/results:/results ${EXTRA} ${DOCKER_IMAGE}
