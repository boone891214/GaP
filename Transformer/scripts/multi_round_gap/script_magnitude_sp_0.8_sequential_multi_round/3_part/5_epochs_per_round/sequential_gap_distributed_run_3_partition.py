import os
from time import sleep
import subprocess
import pipes



epoch_per_train=5
extra_cmd = "" # --short-train  --no-checkpoints


docker_cmd = "nvidia-docker run --rm --ipc=host -v /nasmnt/xiaolong.ma/nv_Transformer/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v /nasmnt/xiaolong.ma/nv_Transformer:/home -v /nasmnt/xiaolong.ma/nv_Transformer/results:/results -w /home transformer"


# first train a blk-1 dense and blk 2,3 sparse model

cmd1 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/1_round/1_grow_1st_blk/ '--sp-retrain --retrain-mask-pattern random --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_sequential_gap_sp_0.8/2_3_0.8_1_dense.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, extra_cmd)

if not os.path.exists('./results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/1_round/1_grow_1st_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, epoch_per_train)):
    os.system(cmd1)

wait = True
while wait:
    if os.path.exists('./results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/1_round/1_grow_1st_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, epoch_per_train)):
        wait = False
        print("wait is over, 1st round 1st model with en dense and de sparse model trained...")
    sleep(10)


# load a model
for round in range(1, 11):

    if round == 1:
        cmd2 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/2_prune_1st_grow_2rd_blk/ '--resume results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/1_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/3_step_sequential_gap_sp_0.8/1_0.8_1_3_masked.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)
    else:
        cmd2 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/2_prune_1st_grow_2rd_blk/ '--resume results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/4_prune_3rd_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/3_step_sequential_gap_sp_0.8/1_0.8_1_3_masked.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, round, epoch_per_train, round - 1, extra_cmd)


    cmd3 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/3_prune_2nd_grow_3rd_blk/ '--resume results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/2_prune_1st_grow_2rd_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/3_step_sequential_gap_sp_0.8/2_0.8_1_2_masked.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)

    cmd4 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/4_prune_3rd_grow_1st_blk/ '--resume results/multi_round_3_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/3_prune_2nd_grow_3rd_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/3_step_sequential_gap_sp_0.8/3_0.8_2_3_masked.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)


    os.system(cmd2)
    os.system(cmd3)
    os.system(cmd4)


    sleep(10)

    
