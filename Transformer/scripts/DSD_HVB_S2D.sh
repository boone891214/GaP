# synchronize files to /workspace/translation/
cp /home/*.py /workspace/translation/ ; \
cp /home/*.sh /workspace/translation/ ; \
cp /home/scripts/*.py /workspace/translation/scripts/ ; \
cp /home/scripts/*.sh /workspace/translation/scripts/ ; \
mkdir -p /workspace/translation/prune_utils/  ; \
cp /home/prune_utils/*.py /workspace/translation/prune_utils/ ; \
cp /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/ ; \
cp /home/fairseq/*.py  /workspace/translation/fairseq/ ; \
cp -r /home/profiles /workspace/translation/;

NUM_DEVICE=${1:-'8'}
VISIBLE_DEVICES=${2:-'0,1,2,3,4,5,6,7'}
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}

######################################################
SEED=2
CURRENT_TIME=`date +"%Y-%m-%d-%T"`
SAVE_FOLDER="/results/NM_HVB-DSD/S2D_fromHVB_ep40_seed${SEED}/"
mkdir -p ${SAVE_FOLDER}
# LOAD_CKPT=${2:-"/home/results/dense_1/checkpoints/checkpoint40.pt"}
# LOAD_CKPT="/data/pretrained/Translation/Transformer/results/sparse_M_N/4_2_H_V_balanced/checkpoints/checkpoint_best.pt"
LOAD_CKPT=/results/oneshot_HVB42_ep40_seed${SEED}/checkpoints/checkpoint_best.pt


# bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 40 5120 ${NUM_DEVICE} ${SAVE_FOLDER} "--resume ${LOAD_CKPT} --restart-training " \
# 2>&1 | tee -a ${SAVE_FOLDER}train_${CURRENT_TIME}.log

# wait
sacrebleu -t wmt14/full -l en-de --echo src | python inference.py --buffer-size 5000 --path ${SAVE_FOLDER}checkpoints/checkpoint_best.pt --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/wmt14_en_de_joined_dict/code --fp16 ; cat results.txt | sacrebleu -t wmt14/full -l en-de -lc \
2>&1 | tee -a ${SAVE_FOLDER}train_${CURRENT_TIME}.log
wait

######################################################
# SEED=2
# CURRENT_TIME=`date +"%Y-%m-%d-%T"`
# SAVE_FOLDER="/results/NM_HVB-DSD/S2D_from42sparsity_ep40_seed${SEED}/"
# mkdir -p ${SAVE_FOLDER}
# # LOAD_CKPT=${2:-"/home/results/dense_1/checkpoints/checkpoint40.pt"}
# # LOAD_CKPT="/data/pretrained/Translation/Transformer/results/sparse_M_N/4_2_H_V_balanced/checkpoints/checkpoint_best.pt"
# LOAD_CKPT=/results/oneshot_42sparsity_ep40_seed${SEED}/checkpoints/checkpoint_best.pt


# bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 40 5120 ${NUM_DEVICE} ${SAVE_FOLDER} "--resume ${LOAD_CKPT} --restart-training " \
# 2>&1 | tee -a ${SAVE_FOLDER}train_${CURRENT_TIME}.log

# wait
# sacrebleu -t wmt14/full -l en-de --echo src | python inference.py --buffer-size 5000 --path ${SAVE_FOLDER}checkpoints/checkpoint_best.pt --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/wmt14_en_de_joined_dict/code --fp16 ; cat results.txt | sacrebleu -t wmt14/full -l en-de -lc \
# 2>&1 | tee -a ${SAVE_FOLDER}train_${CURRENT_TIME}.log
# wait