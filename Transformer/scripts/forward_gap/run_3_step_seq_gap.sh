# synchronize files to /workspace/translation/
cp /home/*.py /workspace/translation/ ; \
cp /home/*.sh /workspace/translation/ ; \
cp -r /home/scripts/* /workspace/translation/scripts/ ; \
mkdir -p /workspace/translation/prune_utils/  ; \
cp /home/prune_utils/*.py /workspace/translation/prune_utils/ ; \
cp -r /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/ ; \
cp /home/fairseq/*.py  /workspace/translation/fairseq/ ;

EPOCH=20
FINAL_EPOCH=40
SPARSITY=0.9
WORKSPACE=/results/GaP/3_step_forward_gap/${SPARSITY}/2nd_round/
LOG_FILE=${WORKSPACE}/ep_${EPOCH}_${FINAL_EPOCH}.txt
START_CKPT=/results/GaP/3_step_forward_gap/0.9/0.9_0.9_0.9/checkpoints/checkpoint_best.pt

rm -f $LOG_FILE

SAVE_FOLDER_STEP_1=${WORKSPACE}/dense_${SPARSITY}_${SPARSITY}/ ; mkdir -p ${SAVE_FOLDER_STEP_1} ; \
bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 ${SAVE_FOLDER_STEP_1} \
"--restart-training --resume ${START_CKPT} --sp-retrain  --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_forward_gap/${SPARSITY}/dense_${SPARSITY}_${SPARSITY}.yaml  " \
2>&1 | tee -a ${LOG_FILE}

# if first round
#"--sp-retrain --retrain-mask-pattern=random --retrain-mask-seed=0 --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_forward_gap/${SPARSITY}/dense_${SPARSITY}_${SPARSITY}.yaml " \


SAVE_FOLDER_STEP_2=${WORKSPACE}/${SPARSITY}_dense_${SPARSITY}/ ; mkdir -p ${SAVE_FOLDER_STEP_2} ; \
LOAD_CKPT=${SAVE_FOLDER_STEP_1}/checkpoints/checkpoint${EPOCH}.pt
bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 ${SAVE_FOLDER_STEP_2} \
"--restart-training --resume ${LOAD_CKPT} --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_forward_gap/${SPARSITY}/${SPARSITY}_dense_${SPARSITY}.yaml  " \
2>&1 | tee -a ${LOG_FILE}

SAVE_FOLDER_STEP_3=${WORKSPACE}/${SPARSITY}_${SPARSITY}_dense/ ; mkdir -p ${SAVE_FOLDER_STEP_3} ; \
LOAD_CKPT=${SAVE_FOLDER_STEP_2}/checkpoints/checkpoint${EPOCH}.pt
bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 ${SAVE_FOLDER_STEP_3} \
"--restart-training --resume ${LOAD_CKPT} --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_forward_gap/${SPARSITY}/${SPARSITY}_${SPARSITY}_dense.yaml  " \
2>&1 | tee -a ${LOG_FILE}


SAVE_FOLDER_STEP_4=${WORKSPACE}/${SPARSITY}_${SPARSITY}_${SPARSITY}/ ; mkdir -p ${SAVE_FOLDER_STEP_4} ; \
LOAD_CKPT=${SAVE_FOLDER_STEP_3}/checkpoints/checkpoint${EPOCH}.pt
bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${FINAL_EPOCH} 5120 8 ${SAVE_FOLDER_STEP_4} \
"--restart-training --resume ${LOAD_CKPT} --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_forward_gap/${SPARSITY}/${SPARSITY}_${SPARSITY}_${SPARSITY}.yaml  " \
2>&1 | tee -a ${LOG_FILE}
