# synchronize files to /workspace/translation/
cp /home/*.py /workspace/translation/ ; \
cp /home/*.sh /workspace/translation/ ; \
cp /home/scripts/*.py /workspace/translation/scripts/ ; \
cp /home/scripts/*.sh /workspace/translation/scripts/ ; \
mkdir -p /workspace/translation/prune_utils/  ; \
cp /home/prune_utils/*.py /workspace/translation/prune_utils/ ; \
cp -r /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/ ; \
cp /home/fairseq/*.py  /workspace/translation/fairseq/ ;

EPOCH=40
LOG_FILE=/home/logs/random_gap_sp_0.8/p80_70.txt
rm -f $LOG_FILE

SAVE_FOLDER=/results/random_gap_sp_0.8/p80_70/ ; mkdir -p ${SAVE_FOLDER} ; \
bash /home/scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 ${SAVE_FOLDER} \
"--sp-backbone --sp-mask-update-freq 2 --sp-update-init-method zero --sp-retrain --retrain-mask-pattern=random --retrain-mask-seed=0 --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/random_gap_sp_0.8/p80_70.yaml" \
2>&1 | tee -a ${LOG_FILE}

