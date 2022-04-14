

# synchronize files to /workspace/translation/
cp /home/*.py /workspace/translation/ ; \
cp /home/*.sh /workspace/translation/ ; \
cp /home/scripts/*.py /workspace/translation/scripts/ ; \
cp /home/scripts/*.sh /workspace/translation/scripts/ ; \
mkdir -p /workspace/translation/prune_utils/  ; \
cp /home/prune_utils/*.py /workspace/translation/prune_utils/ ; \
cp -r /home/prune_utils/ASP/ /workspace/translation/prune_utils/ ; \
cp /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/ ; \
cp /home/fairseq/*.py  /workspace/translation/fairseq/ ; \
cp -r /home/profiles /workspace/translation/;


SAVE_FOLDER=${1:-"/results/tmp/"}
LOAD_CKPT=${2:-"/pretrained/checkpoints/checkpoint_best.pt"}
SPARSITY_TYPE=${3:-"N:M-prune-pattern"} #or 4:2-H-V-balanced or irregular or block
EP=${4:-"40"}
CONFIG_FILE=${5:-"/home/profiles/all_layer_0.8.yaml"}
PRUNE_ARGS=${6:-"--sp-admm-select-number 2 --sp-admm-pattern-row-sub 1 --sp-admm-pattern-col-sub 4 --no-epoch-checkpoints"} #--no-epoch-checkpoints --sp-admm-block (32,32)
RETRAIN_ARGS=${7:-"--sp-admm-select-number 2 --sp-admm-pattern-row-sub 1 --sp-admm-pattern-col-sub 4 --no-epoch-checkpoints"} #--no-epoch-checkpoints

mkdir -p ${SAVE_FOLDER}
bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EP} 5120 8 ${SAVE_FOLDER}/admm/ "--resume ${LOAD_CKPT} --sp-admm --sp-admm-sparsity-type ${SPARSITY_TYPE} --sp-config-file ${CONFIG_FILE} --restart-training ${PRUNE_ARGS}" 2>&1 | tee ${SAVE_FOLDER}/admm/log.txt


bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EP} 5120 8 ${SAVE_FOLDER}/retrain/ "--resume ${SAVE_FOLDER}/admm/checkpoints/checkpoint_best.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type ${SPARSITY_TYPE} --sp-config-file ${CONFIG_FILE} --restart-training ${RETRAIN_ARGS}" 2>&1 | tee ${SAVE_FOLDER}/retrain/log.txt


# for M:N use
# SPARSITY_TYPE=${3:-'N:M-prune-pattern'} #irregular
# CONFIG_FILE=${4:-'/home/profiles/all_layer_0.8.yaml'}
# PRUNE_ARGS=${5:"--sp-admm-select-number 2 --sp-admm-pattern-row-sub 1 --sp-admm-pattern-col-sub 4"}

# for block, use
# SPARSITY_TYPE=${3:-'block'} #irregular
# CONFIG_FILE=${4:-'/home/profiles/all_layer_0.8.yaml'}
# PRUNE_ARGS=${5:"--sp-admm-block='(32,16)' "}

# for HVB, use 4:2-H-V-balanced
