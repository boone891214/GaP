import os
from time import sleep
import subprocess
import pipes
import sys




epoch_per_train=5
extra_cmd = "" # --short-train  --no-checkpoints


docker_cmd = "nvidia-docker run --rm --ipc=host -v /nasmnt/xiaolong.ma/nv_Transformer/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v /nasmnt/xiaolong.ma/nv_Transformer:/home -v /nasmnt/xiaolong.ma/nv_Transformer/results:/results -w /home transformer"


cmd0 = "ssh xiaolong.ma@47.114.139.250 \"nvidia-docker run --rm --ipc=host -v /nasmnt/xiaolong.ma/nv_Transformer/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v /nasmnt/xiaolong.ma/nv_Transformer:/home -v /nasmnt/xiaolong.ma/nv_Transformer/results:/results -w /home transformer bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0 4000 1 5120 8 /results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/0_round/3_combine_all_blk/ '--sp-retrain --retrain-mask-pattern random --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/2_step_parallel_gap_sp_0.8/all_layer_0.8_masked.yaml --restart-training --short-train' \" ".format(epoch_per_train)


if not os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/0_round/3_combine_all_blk/checkpoints/checkpoint_last.pt'.format(epoch_per_train)):
    os.system(cmd0)

wait = True
while wait:
    if os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/0_round/3_combine_all_blk/checkpoints/checkpoint_last.pt'.format(epoch_per_train)):
        wait = False
        print("wait is over, init model created...")
    sleep(10)


bash_cmd = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8".format(epoch_per_train)

# load a model
for round in range(3, 11):
    cmd1 = "ssh xiaolong.ma@47.114.139.250 \"{} {} /results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/1_grow_1st_blk/ '--resume /home/results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/3_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/2_step_parallel_gap_sp_0.8/en_dense_de_masked.yaml --restart-training {}' \"&  ".format(docker_cmd, bash_cmd, epoch_per_train, round, epoch_per_train, round-1, extra_cmd)

    cmd2 = "ssh xiaolong.ma@47.111.236.100 \"{} {} /results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/2_grow_2nd_blk/ ' --resume /home/results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/3_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/2_step_parallel_gap_sp_0.8/de_dense_en_masked.yaml --restart-training {}' \"&  ".format(docker_cmd, bash_cmd, epoch_per_train, round, epoch_per_train, round-1, extra_cmd)


    os.system(cmd1)
    os.system(cmd2)


    wait = True
    while wait:
        if os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/1_grow_1st_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, round, epoch_per_train)) and  os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/2_grow_2nd_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, round, epoch_per_train)):
            wait = False
            print("wait is over, all blk finished...")
        sleep(10)

    cmd_combine = "ssh xiaolong.ma@47.114.139.250 \"{} {} /results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/3_combine_all_blk/ '--sp-retrain --sp-admm-sparsity-type=irregular --sp-prune-before-retrain --sp-config-file=/home/profiles/2_step_parallel_gap_sp_0.8/all_layer_0.8_masked.yaml --load-parallel-model --combine-without-finetune --parallel-1-dir=/home/results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/1_grow_1st_blk/checkpoints/checkpoint_last.pt --parallel-2-dir=/home/results/multi_round_2_partition_parallel_gap_sp_0.8_ep_{}_per_train_no_finetune/{}_round/2_grow_2nd_blk/checkpoints/checkpoint_last.pt {} ' \" ".format(docker_cmd, bash_cmd, epoch_per_train, round, epoch_per_train, round, epoch_per_train, round, extra_cmd)

    os.system(cmd_combine)
    sleep(10)
