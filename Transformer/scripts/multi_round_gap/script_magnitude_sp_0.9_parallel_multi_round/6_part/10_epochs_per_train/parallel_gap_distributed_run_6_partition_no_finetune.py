import os
from time import sleep
import subprocess
import pipes
import sys




epoch_per_train=10
extra_cmd = "" # --short-train  --no-checkpoints


docker_cmd = "nvidia-docker run --rm --ipc=host -v /nasmnt/xiaolong.ma/nv_Transformer/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v /nasmnt/xiaolong.ma/nv_Transformer:/home -v /nasmnt/xiaolong.ma/nv_Transformer/results:/results -w /home transformer"


cmd0 = "ssh xiaolong.ma@47.110.13.43 \"nvidia-docker run --rm --ipc=host -v /nasmnt/xiaolong.ma/nv_Transformer/data/wmt14_en_de_joined_dict:/data/wmt14_en_de_joined_dict -v /nasmnt/xiaolong.ma/nv_Transformer:/home -v /nasmnt/xiaolong.ma/nv_Transformer/results:/results -w /home transformer bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0 4000 1 5120 8 /results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/0_round/7_combine_all_blk/ '--sp-retrain --retrain-mask-pattern random --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/6_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training --short-train' \" ".format(epoch_per_train)


if not os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/0_round/7_combine_all_blk/checkpoints/checkpoint_last.pt'.format(epoch_per_train)):
    os.system(cmd0)

wait = True
while wait:
    if os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/0_round/7_combine_all_blk/checkpoints/checkpoint_last.pt'.format(epoch_per_train)):
        wait = False
        print("wait is over, init model created...")
    sleep(10)


bash_cmd = "bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 {} 5120 8".format(epoch_per_train)

# load a model
for round in range(1, 11):
    cmd1 = "ssh xiaolong.ma@47.110.13.43 \"{} {} /results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/1_grow_1st_blk/ '--resume /home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/7_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/6_step_parallel_gap_sp_0.9/en_0_1_dense_rest_masked.yaml --restart-training {}' \"&  ".format(docker_cmd, bash_cmd, epoch_per_train, round, epoch_per_train, round-1, extra_cmd)

    cmd2 = "ssh xiaolong.ma@47.114.139.250 \"{} {} /results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/2_grow_2nd_blk/ ' --resume /home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/7_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/6_step_parallel_gap_sp_0.9/en_2_3_dense_rest_masked.yaml --restart-training {}' \"&  ".format(docker_cmd, bash_cmd, epoch_per_train, round, epoch_per_train, round-1, extra_cmd)

    cmd3 = "ssh xiaolong.ma@47.111.236.100 \"{} {} /results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/3_grow_3rd_blk/ ' --resume /home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/7_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/6_step_parallel_gap_sp_0.9/en_4_5_dense_rest_masked.yaml --restart-training {}' \"&  ".format(docker_cmd, bash_cmd, epoch_per_train, round, epoch_per_train, round-1, extra_cmd)

    cmd4 = "ssh xiaolong.ma@121.196.121.123 \"{} {} /results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/4_grow_4th_blk/ ' --resume /home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/7_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/6_step_parallel_gap_sp_0.9/de_0_1_dense_rest_masked.yaml --restart-training {}' \"&  ".format(docker_cmd, bash_cmd, epoch_per_train, round, epoch_per_train, round-1, extra_cmd)

    cmd5 = "ssh xiaolong.ma@47.114.150.219 \"{} {} /results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/5_grow_5th_blk/ ' --resume /home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/7_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/6_step_parallel_gap_sp_0.9/de_2_3_dense_rest_masked.yaml --restart-training {}' \"&  ".format(docker_cmd, bash_cmd, epoch_per_train, round, epoch_per_train, round-1, extra_cmd)

    cmd6 = "{} {} /results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/6_grow_6th_blk/ ' --resume /home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/7_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/6_step_parallel_gap_sp_0.9/de_4_5_dense_rest_masked.yaml --restart-training {}' &  ".format(docker_cmd, bash_cmd, epoch_per_train, round, epoch_per_train, round-1, extra_cmd)


    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)
    os.system(cmd4)
    os.system(cmd5)
    os.system(cmd6)


    wait = True
    while wait:
        if os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/1_grow_1st_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, round, epoch_per_train)) and  os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/2_grow_2nd_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, round, epoch_per_train)) and  os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/3_grow_3rd_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, round, epoch_per_train)) and os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/4_grow_4th_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, round, epoch_per_train)) and os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/5_grow_5th_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, round, epoch_per_train)) and os.path.exists('/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/6_grow_6th_blk/checkpoints/checkpoint{}.pt'.format(epoch_per_train, round, epoch_per_train)):
            wait = False
            print("wait is over, all blk finished...")
        sleep(10)

    cmd_combine = "ssh xiaolong.ma@121.196.121.123 \"{} {} /results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/7_combine_all_blk/ '--sp-retrain --sp-admm-sparsity-type=irregular --sp-prune-before-retrain --sp-config-file=/home/profiles/6_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --load-parallel-model --combine-without-finetune --parallel-1-dir=/home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/1_grow_1st_blk/checkpoints/checkpoint_last.pt --parallel-2-dir=/home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/2_grow_2nd_blk/checkpoints/checkpoint_last.pt --parallel-3-dir=/home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/3_grow_3rd_blk/checkpoints/checkpoint_last.pt --parallel-4-dir=/home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/4_grow_4th_blk/checkpoints/checkpoint_last.pt --parallel-5-dir=/home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/5_grow_5th_blk/checkpoints/checkpoint_last.pt --parallel-6-dir=/home/results/multi_round_6_partition_parallel_gap_sp_0.9_ep_{}_per_train_no_finetune/{}_round/6_grow_6th_blk/checkpoints/checkpoint_last.pt ' \" ".format(docker_cmd, bash_cmd, epoch_per_train, round, epoch_per_train, round, epoch_per_train, round, epoch_per_train, round, epoch_per_train, round, epoch_per_train, round, epoch_per_train, round)

    os.system(cmd_combine)
    sleep(10)
