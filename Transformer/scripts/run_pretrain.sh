

# synchronize files to /workspace/translation/
cp /home/*.py /workspace/translation/ ; \
cp /home/*.sh /workspace/translation/ ; \
cp /home/scripts/*.py /workspace/translation/scripts/ ; \
cp /home/scripts/*.sh /workspace/translation/scripts/ ; \
mkdir -p /workspace/translation/prune_utils/  ; \
cp /home/prune_utils/*.py /workspace/translation/prune_utils/ ; \
cp -r /home/prune_utils/ASP/ /workspace/translation/prune_utils/ ; \
cp /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/ ; \
cp /home/fairseq/models/*.py /workspace/translation/fairseq/models/ ; \
cp /home/fairseq/*.py  /workspace/translation/fairseq/ ; \
cp -r /home/profiles /workspace/translation/;


EXTRA_ARGS=${1:-""}
SAVE_FOLDER=${2:-"/results/tmp/"}
LOAD_CKPT=${3:-"None"}
EMB_DIM=${4:-"1024"}
FF_EMB_DIM=${5:-"4096"}
NUM_EN_H=${6:-"16"}
NUM_DE_H=${7:-"16"}
EP=${8:-"40"}
NUM_GPU=${9:-"8"}
INIT_LR=${10:-"0.000846"}
SEED=${11:-"1"}
AMP=${12-"amp"}


mkdir -p ${SAVE_FOLDER}
bash scripts/run_DGX1_AMP_8GPU.sh ${AMP} ${SEED} ${INIT_LR} 4000 ${EP} 5120 ${NUM_GPU} ${SAVE_FOLDER} transformer_wmt_en_de_big_t2t "--resume ${LOAD_CKPT} --restart-training --encoder-embed-dim=$EMB_DIM --decoder-embed-dim=$EMB_DIM --encoder-ffn-embed-dim=${FF_EMB_DIM} --decoder-ffn-embed-dim=${FF_EMB_DIM} --encoder-attention-heads ${NUM_EN_H} --decoder-attention-heads ${NUM_DE_H} --no-epoch-checkpoints ${EXTRA_ARGS}"  2>&1 | tee ${SAVE_FOLDER}/log.txt
