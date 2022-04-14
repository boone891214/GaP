# synchronize files to /workspace/translation/
cp /home/*.py /workspace/translation/ ; \
cp /home/*.sh /workspace/translation/ ; \
cp /home/scripts/*.py /workspace/translation/scripts/ ; \
cp /home/scripts/*.sh /workspace/translation/scripts/ ; \
mkdir -p /workspace/translation/prune_utils/  ; \
cp /home/prune_utils/*.py /workspace/translation/prune_utils/ ; \
cp -r /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/ ; \
cp /home/fairseq/*.py  /workspace/translation/fairseq/ ;

EPOCH=30
FINAL_EPOCH=40
LOG_FILE=/home/logs/2_step_seq_gap_ep_${EPOCH}_${FINAL_EPOCH}.txt
rm -f $LOG_FILE

SAVE_FOLDER=/results/seq_gap/encoder_0_decoder_0.8/ ; mkdir -p ${SAVE_FOLDER} ; \
bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 ${SAVE_FOLDER} \
"--sp-retrain --retrain-mask-pattern=random --retrain-mask-seed=0 --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/2_step_forward_gap/encoder_0_decoder_0.8.yaml" \
2>&1 | tee -a ${LOG_FILE}

SAVE_FOLDER=/results/seq_gap/encoder_0.8_decoder_0/ ; mkdir -p ${SAVE_FOLDER} ; \
LOAD_CKPT=/results/seq_gap/encoder_0_decoder_0.8/checkpoints/checkpoint${EPOCH}.pt
bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 ${SAVE_FOLDER} \
"--restart-training --resume ${LOAD_CKPT} --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/2_step_forward_gap/encoder_0.8_decoder_0.yaml" \
2>&1 | tee -a ${LOG_FILE}

SAVE_FOLDER=/results/seq_gap/encoder_0.8_decoder_0.8/ ; mkdir -p ${SAVE_FOLDER} ; \
LOAD_CKPT=/results/seq_gap/encoder_0.8_decoder_0/checkpoints/checkpoint${EPOCH}.pt
bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${FINAL_EPOCH} 5120 8 ${SAVE_FOLDER} \
"--restart-training --resume ${LOAD_CKPT} --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/2_step_forward_gap/encoder_masked_decoder_0.8.yaml" \
2>&1 | tee -a ${LOG_FILE}
