# synchronize files to /workspace/translation/
cp /home/*.py /workspace/translation/ ; \
cp /home/*.sh /workspace/translation/ ; \
cp /home/scripts/*.py /workspace/translation/scripts/ ; \
cp /home/scripts/*.sh /workspace/translation/scripts/ ; \
mkdir -p /workspace/translation/prune_utils/  ; \
cp /home/prune_utils/*.py /workspace/translation/prune_utils/ ; \
cp -r /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/ ; \
cp /home/fairseq/*.py  /workspace/translation/fairseq/ ;

NUM_DEVICE=${1:-'8'}
VISIBLE_DEVICES=${2:-'0,1,2,3,4,5,6,7'}
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}

CURRENT_TIME=`date +"%Y-%m-%d-%T"`
EPOCH=70 #40
MASK_UPDATE_FREQ=${3:-"2"}
SAVE_FOLDER=/results/random_gap_HVB42/ep${EPOCH}_maskUpdate${MASK_UPDATE_FREQ}/ ; mkdir -p ${SAVE_FOLDER} ; \
# LOG_FILE=${SAVE_FOLDER}log_${CURRENT_TIME}.log
# SP_CONFIG_FILE='/home/profiles/random_gap_sp_NM42_balance/p0.yaml'
SP_CONFIG_FILE=/home/profiles/random_gap_sp_NM42_balance/random_gap_HVB_ep50.yaml # ep50 for random gap + 20 ep for finetune
bash /home/scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 ${NUM_DEVICE} ${SAVE_FOLDER} \
"--sp-backbone --sp-mask-update-freq 2 --sp-update-init-method zero --sp-retrain --sp-admm-sparsity-type 4:2-H-V-balanced --sp-prune-before-retrain --sp-config-file ${SP_CONFIG_FILE}" \
2>&1 | tee -a ${SAVE_FOLDER}train_${CURRENT_TIME}.log

# #---- evaluation ---#
LOAD_CKPT=${SAVE_FOLDER}checkpoints/checkpoint_last.pt
sacrebleu -t wmt14/full -l en-de --echo src | python inference.py --buffer-size 5000 --path ${LOAD_CKPT} --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/wmt14_en_de_joined_dict/code --fp16; cat results.txt | sacrebleu -t wmt14/full -l en-de -lc \
2>&1 | tee -a ${SAVE_FOLDER}eval_${CURRENT_TIME}.log