set -x

# synchronize files to /workspace/translation/
cp /home/*.py /workspace/translation/ ; \
cp /home/*.sh /workspace/translation/ ; \
cp -rf /home/scripts/ /workspace/translation/scripts/ ; \
mkdir -p /workspace/translation/prune_utils/  ; \
cp /home/prune_utils/*.py /workspace/translation/prune_utils/ ; \
cp -r /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/ ; \
cp /home/fairseq/*.py  /workspace/translation/fairseq/ ; \
cp -rf /home/profiles /workspace/translation/profiles/ ;


CUDA_DEVICES=${1:-"0,1,2,3"}
NUM_DEVICES=${2:-"4"}
OUT_DIR=${3:-"/results/"}
CONFIG=${4:-""}
EPOCHS=${5:-"40"}

EXTRA_ARGS=${6:-""}

OUT_DIR="/results/${OUT_DIR}"
rm -rf ${OUT_DIR}
mkdir -p ${OUT_DIR}
mkdir -p ${OUT_DIR}/checkpoints

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} bash /home/scripts/run_training_fsun.sh 'amp' 1 0.000846 4000 ${EPOCHS} 5120 ${NUM_DEVICES} ${OUT_DIR} \
"--sp-backbone --sp-mask-update-freq 1 --sp-update-init-method zero --sp-retrain --retrain-mask-pattern=random --retrain-mask-seed=0 --sp-admm-sparsity-type irregular --sp-config-file ${CONFIG} ${EXTRA_ARGS}" \
2>&1 | tee -a ${OUT_DIR}/train.log
