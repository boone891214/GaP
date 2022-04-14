BLOCK_SIZE=${1:-'2,2'}
NUM_DEVICE=${2:-'8'}
VISIBLE_DEVICE=${3:-'0,1,2,3,4,5,6,7'}
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICE}
CURRENT_TIME=`date +"%Y-%m-%d-%T"`
PRUNE_ARGS="--sp-admm-block=(${BLOCK_SIZE}) "
#---- trainig, sparsity = 0.9 ---#
SPARSITY_TYPE='block' #irregular
CONFIG_FILE='/home/profiles/all_layer_0.9.yaml'
SAVE_FOLDER="/results/block_sparse_sp_0.9_${BLOCK_SIZE}/"
LOAD_CKPT_LAST="/results/block_sparse_sp_0.8_${BLOCK_SIZE}/checkpoints/checkpoint_best.pt"

# bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 40 5120 ${NUM_DEVICE} ${SAVE_FOLDER} "--resume ${LOAD_CKPT_LAST} --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type ${SPARSITY_TYPE} --sp-config-file ${CONFIG_FILE} --restart-training ${PRUNE_ARGS}" \
# 2>&1 | tee -a ${SAVE_FOLDER}train_${CURRENT_TIME}.log

# #---- evaluation, sparsity = 0.9 ---#
LOAD_CKPT=${SAVE_FOLDER}checkpoints/checkpoint_best.pt
sacrebleu -t wmt14/full -l en-de --echo src | python inference.py --buffer-size 5000 --path ${LOAD_CKPT} --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/wmt14_en_de_joined_dict/code --fp16; cat results.txt | sacrebleu -t wmt14/full -l en-de -lc \
2>&1 | tee -a ${SAVE_FOLDER}eval_${CURRENT_TIME}.log