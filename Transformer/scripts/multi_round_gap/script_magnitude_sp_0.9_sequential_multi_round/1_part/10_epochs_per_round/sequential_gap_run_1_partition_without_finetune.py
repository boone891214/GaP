import os
from time import sleep
import subprocess
import pipes



epoch_per_train=10
extra_cmd = "" # --short-train  --no-checkpoints


docker_cmd = "nvidia-docker run --rm --ipc=host -v /nasmnt/xiaolong.ma/nv_Transformer/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v /nasmnt/xiaolong.ma/nv_Transformer:/home -v /nasmnt/xiaolong.ma/nv_Transformer/results:/results -w /home transformer"


# first train a all dense model

cmd1 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_1_partition_sequential_gap_sp_0.9_ep_{}_per_train_no_finetune/1_round/1_grow_all_blk/ ".format(epoch_per_train, epoch_per_train)

if not os.path.exists('./results/multi_round_1_partition_sequential_gap_sp_0.9_ep_{}_per_train_no_finetune/1_round/1_grow_all_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, epoch_per_train)):
    os.system(cmd1)

wait = True
while wait:
    if os.path.exists('./results/multi_round_1_partition_sequential_gap_sp_0.9_ep_{}_per_train_no_finetune/1_round/1_grow_all_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, epoch_per_train)):
        wait = False
        print("wait is over, 1st round 1st model with en dense and de sparse model trained...")
    sleep(10)


# load a model
for round in range(1, 11):

    if round == 1:
        cmd2 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_1_partition_sequential_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/2_prune_all_grow_all_blk/ '--resume results/multi_round_1_partition_sequential_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/1_grow_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/1_step_sequential_gap_sp_0.9/all_0.9_no_masked.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, round, epoch_per_train, round, extra_cmd)
    else:
        cmd2 = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8 /results/multi_round_1_partition_sequential_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/2_prune_all_grow_all_blk/ '--resume results/multi_round_1_partition_sequential_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/2_prune_all_grow_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-admm-sparsity-type irregular --sp-prune-before-retrain --sp-config-file /home/profiles/1_step_sequential_gap_sp_0.9/all_0.9_no_masked.yaml --restart-training {}' ".format(epoch_per_train, epoch_per_train, round, epoch_per_train, round - 1, extra_cmd)


    os.system(cmd2)


    sleep(10)

    
