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
SEED=${3:-"1"}
BSIZE=${4:-"5120"}
MODEL_ARCH=transformer_wmt_en_de_big_t2t_LargerEmbed
SAVE_FOLDER_TAIL=_seed${SEED}
CURRENT_TIME=`date +"%Y-%m-%d-%T"`
EPOCH=40
LOAD_CKPT=/results/${MODEL_ARCH}_dense_ep40_seed${SEED}/checkpoints/checkpoint_best.pt

###################################################################
# EPOCH=40
LRATE=0.000423 # dense=0.000846
SAVE_FOLDER=/results/${MODEL_ARCH}_oneshot_42sparsity_ep${EPOCH}_LR${LRATE}_${SAVE_FOLDER_TAIL}/
mkdir -p ${SAVE_FOLDER}
SP_CONFIG_FILE=/home/profiles/all_layer_0.5.yaml # ep50 for random gap + 20 ep for finetune

bash /home/scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 ${LRATE} 4000 ${EPOCH} ${BSIZE} ${NUM_DEVICE} ${SAVE_FOLDER} ${MODEL_ARCH} \
"--resume ${LOAD_CKPT} --restart-training --sp-backbone --sp-retrain --sp-admm-sparsity-type N:M-prune-pattern --sp-update-init-method zero --sp-admm-pattern-col-sub 4 --sp-admm-pattern-row-sub 1 --sp-admm-select-number 2 --sp-prune-before-retrain --sp-config-file ${SP_CONFIG_FILE}" \
2>&1 | tee -a ${SAVE_FOLDER}train_${CURRENT_TIME}.log
wait
# #---- evaluation ---#
LOAD_CKPT=${SAVE_FOLDER}checkpoints/checkpoint_best.pt
sacrebleu -t wmt14/full -l en-de --echo src | python inference.py --buffer-size 5000 --path ${LOAD_CKPT} --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/wmt14_en_de_joined_dict/code --fp16; cat results.txt | sacrebleu -t wmt14/full -l en-de -lc \
2>&1 | tee -a ${SAVE_FOLDER}eval_${CURRENT_TIME}.log
##################################################################
