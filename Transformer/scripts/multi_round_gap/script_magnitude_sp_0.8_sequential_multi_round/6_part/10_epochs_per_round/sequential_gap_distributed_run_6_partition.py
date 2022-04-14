import os
from time import sleep
import subprocess
import pipes



epoch_per_train=10
extra_cmd = "" # --short-train  --no-checkpoints


docker_cmd = "nvidia-docker run --rm --ipc=host -v /nasmnt/xiaolong.ma/nv_Transformer/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v /nasmnt/xiaolong.ma/nv_Transformer:/home -v /nasmnt/xiaolong.ma/nv_Transformer/results:/results -w /home transformer"


# first train a blk-1 dense and blk 2,3,4 sparse model

cmd1 = "{} bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/1_round/1_grow_1st_blk/ '--sp-retrain --retrain-mask-pattern random --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/6_step_sequential_gap_sp_0.8/blk_23456_0.8_blk_1_dense.yaml --restart-training {}' ".format(docker_cmd, epoch_per_train, epoch_per_train, extra_cmd)

if not os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/1_round/1_grow_1st_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, epoch_per_train)):
    os.system(cmd1)

wait = True
while wait:
    if os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/1_round/1_grow_1st_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, epoch_per_train)):
        wait = False
        print("wait is over, 1st round 1st model with blk-1 dense and blk 2,3,4,5,6 sparse model trained...")
    sleep(10)


# load a model
for round in range(3, 11):

    if round == 1:
        cmd2 = "{} bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/2_prune_1st_grow_2rd_blk/ '--resume results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/1_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/6_step_sequential_gap_sp_0.8/blk_1_0.8_blk_13456_masked.yaml --restart-training {}' ".format(docker_cmd, epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)
    else:
        cmd2 = "{} bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/2_prune_1st_grow_2rd_blk/ '--resume results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/7_prune_6th_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/6_step_sequential_gap_sp_0.8/blk_1_0.8_blk_13456_masked.yaml --restart-training {}' ".format(docker_cmd, epoch_per_train, epoch_per_train, round, epoch_per_train, round - 1, extra_cmd)


    cmd3 = "{} bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/3_prune_2rd_grow_3nd_blk/ '--resume results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/2_prune_1st_grow_2rd_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/6_step_sequential_gap_sp_0.8/blk_2_0.8_blk_12456_masked.yaml --restart-training {}' ".format(docker_cmd, epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)

    cmd4 = "{} bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/4_prune_3nd_grow_4th_blk/ '--resume results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/3_prune_2rd_grow_3nd_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/6_step_sequential_gap_sp_0.8/blk_3_0.8_blk_12356_masked.yaml --restart-training {}' ".format(docker_cmd, epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)

    cmd5 = "{} bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/5_prune_4th_grow_5th_blk/ '--resume results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/4_prune_3nd_grow_4th_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/6_step_sequential_gap_sp_0.8/blk_4_0.8_blk_12346_masked.yaml --restart-training {}' ".format(docker_cmd, epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)

    cmd6 = "{} bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/6_prune_5th_grow_6th_blk/ '--resume results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/5_prune_4th_grow_5th_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/6_step_sequential_gap_sp_0.8/blk_5_0.8_blk_12345_masked.yaml --restart-training {}' ".format(docker_cmd, epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)

    cmd7 = "{} bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/7_prune_6th_grow_1st_blk/ '--resume results/multi_round_6_partition_sequential_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/6_prune_5th_grow_6th_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/6_step_sequential_gap_sp_0.8/blk_6_0.8_blk_23456_masked.yaml --restart-training {}' ".format(docker_cmd, epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)


    os.system(cmd2)
    os.system(cmd3)
    os.system(cmd4)
    os.system(cmd5)
    os.system(cmd6)
    os.system(cmd7)


    sleep(10)

    
