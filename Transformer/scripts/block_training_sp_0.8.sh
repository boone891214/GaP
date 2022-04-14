BLOCK_SIZE=${1:-'2,2'}
NUM_DEVICE=${2:-'8'}
VISIBLE_DEVICE=${3:-'0,1,2,3,4,5,6,7'}
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICE}
# PRETRAINED_MODEL=/pretrained/checkpoints/checkpoint_best.pt 
# SAVE_FOLDER=/results/sparse_1/ ; mkdir -p ${SAVE_FOLDER} ; \
# bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 40 5120 ${NUM_DEVICE} ${SAVE_FOLDER} "--resume '${PRETRAINED_MODEL}' --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type block --sp-admm-block=\(${BLOCK_SIZE}\) --sp-config-file /home/profiles/all_layer_0.8.yaml --restart-training --checkpoint-dir=./checkpoints/block_${BLOCK_SIZE}/SPARSITY_0.8/"

# bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 40 5120 ${NUM_DEVICE} ${SAVE_FOLDER} "--resume ./checkpoints/block_${BLOCK_SIZE}/SPARSITY_0.8/checkpoint_best.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type block --sp-admm-block=\(${BLOCK_SIZE}\) --sp-config-file /home/profiles/all_layer_0.9.yaml --restart-training --checkpoint-dir=./checkpoints/block_${BLOCK_SIZE}/SPARSITY_0.9/"




# synchronize files to /workspace/translation/
# cp /home/*.py /workspace/translation/ ; \
# cp /home/*.sh /workspace/translation/ ; \
# cp /home/scripts/*.py /workspace/translation/scripts/ ; \
# cp /home/scripts/*.sh /workspace/translation/scripts/ ; \
# mkdir -p /workspace/translation/prune_utils/  ; \
# cp /home/prune_utils/*.py /workspace/translation/prune_utils/ ; \
# cp /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/ ; \
# cp /home/fairseq/*.py  /workspace/translation/fairseq/ ; \
# cp -r /home/profiles /workspace/translation/;



# LOAD_CKPT=${2:-"/home/results/dense_1/checkpoints/checkpoint40.pt"}
LOAD_CKPT_DENSE="/data/pretrained/Translation/Transformer/results/dense_1/checkpoints/checkpoint_best.pt"

# for M:N use
# SPARSITY_TYPE=${3:-'N:M-prune-pattern'} #irregular
# CONFIG_FILE=${4:-'/home/profiles/all_layer_0.8.yaml'}
# PRUNE_ARGS=${5:"--sp-admm-select-number 2 --sp-admm-pattern-row-sub 1 --sp-admm-pattern-col-sub 4"}
# SAVE_FOLDER="/results/tmp/"

# for block, use
#---- trainig, sparsity = 0.8 ---#
CURRENT_TIME=`date +"%Y-%m-%d-%T"`
SPARSITY_TYPE='block' #irregular
PRUNE_ARGS="--sp-admm-block=(${BLOCK_SIZE}) "
CONFIG_FILE='/home/profiles/all_layer_0.8.yaml'

SAVE_FOLDER="/results/block_sparse_sp_0.8_${BLOCK_SIZE}/"

# mkdir -p ${SAVE_FOLDER}
# bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 40 5120 ${NUM_DEVICE} ${SAVE_FOLDER} "--resume ${LOAD_CKPT_DENSE} --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type ${SPARSITY_TYPE} --sp-config-file ${CONFIG_FILE} --restart-training ${PRUNE_ARGS}" \
# 2>&1 | tee -a ${SAVE_FOLDER}train_${CURRENT_TIME}.log


# #---- evaluation, sparsity = 0.8 ---#
LOAD_CKPT=${SAVE_FOLDER}checkpoints/checkpoint_best.pt
sacrebleu -t wmt14/full -l en-de --echo src | python inference.py --buffer-size 5000 --path ${LOAD_CKPT} --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/wmt14_en_de_joined_dict/code --fp16; cat results.txt | sacrebleu -t wmt14/full -l en-de -lc \
2>&1 | tee -a ${SAVE_FOLDER}eval_${CURRENT_TIME}.log