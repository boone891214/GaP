

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


LOAD_CKPT=${1:-"/results/rebuttal_ICLR/step_15/checkpoints/checkpoint_last.pt"}
PREC=${2:-"--fp16"}

cd /workspace/translation/

rm -rf results.txt
sacrebleu -t wmt14/full -l en-de --echo src | python inference.py --buffer-size 5000 --path ${LOAD_CKPT} --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/wmt14_en_de_joined_dict/code ${PREC} ; cat /workspace/translation/results.txt | sacrebleu -t wmt14/full -l en-de -lc

cd /home
