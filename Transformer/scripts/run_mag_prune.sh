

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


SAVE_FOLDER=${1:-"/results/rebuttal_ICLR/finetune_step_15/"}
LOAD_CKPT=${2:-"/home/results/rebuttal_ICLR/step_15/checkpoints/checkpoint_last.pt"}
LR=${3-"0.000846"}
EP=${4:-"40"}
SPARSITY_TYPE=${5:-"irregular"} #or 4:2-H-V-balanced or irregular or block
CONFIG_FILE=${6:-"/home/profiles/all_layer_0.9.yaml"}
PRUNE_ARGS=${7:-"--sp-admm-select-number 2 --sp-admm-pattern-row-sub 1 --sp-admm-pattern-col-sub 4"} #--sp-global-weight-sparsity 0.9 --sp-prune-threshold 0.2 --no-epoch-checkpoints
PREC=${8:-"amp"}

mkdir -p ${SAVE_FOLDER}
bash scripts/run_DGX1_AMP_8GPU.sh ${PREC} 1 ${LR} 4000 ${EP} 5120 8 ${SAVE_FOLDER} transformer_wmt_en_de_big_t2t "--resume ${LOAD_CKPT} --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type ${SPARSITY_TYPE} --sp-config-file ${CONFIG_FILE} --restart-training ${PRUNE_ARGS}" 2>&1 | tee ${SAVE_FOLDER}/log.txt

# for M:N use
# SPARSITY_TYPE=${3:-'N:M-prune-pattern'} #irregular
# CONFIG_FILE=${4:-'/home/profiles/all_layer_0.8.yaml'}
# PRUNE_ARGS=${5:"--sp-admm-select-number 2 --sp-admm-pattern-row-sub 1 --sp-admm-pattern-col-sub 4"}

# for block, use
# SPARSITY_TYPE=${3:-'block'} #irregular
# CONFIG_FILE=${4:-'/home/profiles/all_layer_0.8.yaml'}
# PRUNE_ARGS=${5:"--sp-admm-block='(32,16)' "}

# for HVB, use 4:2-H-V-balanced
