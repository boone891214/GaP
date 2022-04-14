

# synchronize files to /workspace/translation/
cp /home/*.py /workspace/translation/ ; \
cp /home/*.sh /workspace/translation/ ; \
cp /home/scripts/*.py /workspace/translation/scripts/ ; \
cp /home/scripts/*.sh /workspace/translation/scripts/ ; \
mkdir -p /workspace/translation/prune_utils/  ; \
cp /home/prune_utils/*.py /workspace/translation/prune_utils/ ; \
cp /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/ ; \
cp /home/fairseq/models/*.py /workspace/translation/fairseq/models/ ; \
cp /home/fairseq/*.py  /workspace/translation/fairseq/ ; \
cp -r /home/profiles /workspace/translation/;

NUM_GPU=${1:-"8"}
VISIBLE_DEVICES=${2:-'0,1,2,3,4,5,6,7'}
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}
EPOCH=${3:-"40"} #40
SEED=${4:-"1"}
BSIZE=${5:-"5120"}
CURRENT_TIME=`date +"%Y-%m-%d-%T"`

MODEL_ARCH=transformer_wmt_en_de_big_t2t_LargerEmbed
INIT_LR="0.0006768" # 
SAVE_FOLDER="/results/${MODEL_ARCH}_dense_ep${EPOCH}_seed${SEED}_LR${INIT_LR}/"
mkdir -p ${SAVE_FOLDER}
# LOAD_CKPT=${SAVE_FOLDER}"checkpoints/checkpoint40.pt"
# INIT_LR="0.000846"


bash scripts/run_DGX1_AMP_8GPU.sh 'amp' ${SEED} ${INIT_LR} 4000 ${EPOCH} ${BSIZE} ${NUM_GPU} ${SAVE_FOLDER} ${MODEL_ARCH} #"--resume ${LOAD_CKPT} --restart-training "
2>&1 | tee -a ${SAVE_FOLDER}train_${CURRENT_TIME}.log

wait
# #---- evaluation, sparsity = 0.8 ---#
LOAD_CKPT=${SAVE_FOLDER}checkpoints/checkpoint_best.pt
sacrebleu -t wmt14/full -l en-de --echo src | python inference.py --buffer-size 5000 --path ${LOAD_CKPT} --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/wmt14_en_de_joined_dict/code --fp16; cat results.txt | sacrebleu -t wmt14/full -l en-de -lc \
2>&1 | tee -a ${SAVE_FOLDER}eval_${CURRENT_TIME}.log