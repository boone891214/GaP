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
