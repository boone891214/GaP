import os
from time import sleep
import subprocess
import pipes



epoch_per_train=10
extra_cmd = "" # --short-train  --no-checkpoints


docker_cmd = "nvidia-docker run --rm --ipc=host -v /nasmnt/xiaolong.ma/nv_Transformer/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v /nasmnt/xiaolong.ma/nv_Transformer:/home -v /nasmnt/xiaolong.ma/nv_Transformer/results:/results -w /home transformer"


# first train a blk-1 dense and blk 2,3 sparse model

cmd1 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2+1/sequential_1/1_grow_1st_blk/ '--resume results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/3_step_sequential_gap_sp_0.9/para_to_seq_2_3_masked.yaml --restart-training {}' ".format(epoch_per_train, extra_cmd)

cmd2 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2+1/sequential_1/2_prune_1st_grow_2rd_blk/ '--resume results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2+1/sequential_1/1_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/3_step_sequential_gap_sp_0.9/1_0.9_1_3_masked.yaml --restart-training {}' ".format(epoch_per_train, extra_cmd)

cmd3 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2+1/sequential_1/3_prune_2nd_grow_3rd_blk/ '--resume results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2+1/sequential_1/2_prune_1st_grow_2rd_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/3_step_sequential_gap_sp_0.9/2_0.9_1_2_masked.yaml --restart-training {}' ".format(epoch_per_train, extra_cmd)

cmd4 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2+1/sequential_1/4_prune_3rd_grow_1st_blk/ '--resume results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2+1/sequential_1/3_prune_2nd_grow_3rd_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/3_step_sequential_gap_sp_0.9/3_0.9_2_3_masked.yaml --restart-training {}' ".format(epoch_per_train, extra_cmd)


os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
os.system(cmd4)


cmd_finetune = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 40 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2+1/sequential_1_finetune/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2+1/sequential_1/4_prune_3rd_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_sequential_gap_sp_0.9/finetune.yaml --restart-training ' "

os.system(cmd_finetune)