import os
from time import sleep
import subprocess
import pipes



epoch_per_train=10
extra_cmd = "" # --short-train  --no-checkpoints


docker_cmd = "nvidia-docker run --rm --ipc=host -v /nasmnt/xiaolong.ma/nv_Transformer/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v /nasmnt/xiaolong.ma/nv_Transformer:/home -v /nasmnt/xiaolong.ma/nv_Transformer/results:/results -w /home transformer"


# first train a blk-1 dense and blk 2 sparse model

cmd_init = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.00 4000 1 5120 8 /results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/0_round/random_init/ '--restart-training --short-train' ".format(epoch_per_train)

cmd1 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/1_round/1_grow_1st_blk/ '--resume results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/0_round/random_init/checkpoints/checkpoint_last.pt  --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type=block --sp-admm-block=(1,8) --sp-config-file /home/profiles/2_step_sequential_gap_sp_0.8/de_0.8_masked.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, epoch_per_train, extra_cmd)

if not os.path.exists('./results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/1_round/1_grow_1st_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, epoch_per_train-1)):
    os.system(cmd_init)
    os.system(cmd1)

wait = True
while wait:
    if os.path.exists('./results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/1_round/1_grow_1st_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, epoch_per_train-1)):
        wait = False
        print("wait is over, 1st round 1st model with en dense and de sparse model trained...")
    sleep(10)


# load a model
for round in range(1, 11):

    if round == 1:
        cmd2 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/{}_round/2_prune_1st_grow_2rd_blk/ '--resume results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/{}_round/1_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type=block --sp-admm-block=(1,8) --sp-prune-before-retrain --sp-config-file /home/profiles/2_step_sequential_gap_sp_0.8/en_0.8_masked.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)
    else:
        cmd2 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/{}_round/2_prune_1st_grow_2rd_blk/ '--resume results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/{}_round/3_prune_2nd_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type=block --sp-admm-block=(1,8) --sp-prune-before-retrain --sp-config-file /home/profiles/2_step_sequential_gap_sp_0.8/en_0.8_masked.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, round, epoch_per_train, round - 1, extra_cmd)


    cmd3 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/{}_round/3_prune_2nd_grow_1st_blk/ '--resume results/multi_round_2_partition_sequential_gap_sp_0.8_block_1_8_ep_{}_per_train_no_finetune/{}_round/2_prune_1st_grow_2rd_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type=block --sp-admm-block=(1,8) --sp-prune-before-retrain --sp-config-file /home/profiles/2_step_sequential_gap_sp_0.8/de_0.8_masked.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)


    os.system(cmd2)
    os.system(cmd3)


    sleep(10)

    
