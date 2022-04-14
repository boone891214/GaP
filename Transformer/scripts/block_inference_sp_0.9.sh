# #---- evaluation, sparsity = 0.9 ---#
LOAD_CKPT="/results/block_sp_0.9/checkpoint_best.pt"
sacrebleu -t wmt14/full -l en-de --echo src | python inference.py --buffer-size 5000 --path ${LOAD_CKPT} --max-tokens 10240 --fuse-dropout-add --remove-bpe --bpe-codes /data/wmt14_en_de_joined_dict/code --fp16 ; cat results.txt | sacrebleu -t wmt14/full -l en-de -lc  