import os
import time
import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--gpu-mem",default='32510MiB',type=str, help="total memory of a single gpu")
parser.add_argument("--try-sec", default=30, type=int, help="test availablity for every x seconds")
parser.add_argument("--cmd", default=" ", type=str, help="command to run")

args = parser.parse_args()

keywords = [args.gpu_mem, 'Default']

cmd_to_run ="bash scripts/run_pretraining_mag_prune.sh True True results/first_pretrain_dense/checkpoints/ckpt_8601.pt results/mag_prune_pretrain/4-2-nohvb-dsemb-lr0.003-0.002/ \"--nv-sparse --sp-admm-sparsity-type N:M-prune-pattern --print-model --disable_progress_bar\"  true 8192 3e-3 fp16 8 0.2843 7038 100 true true 128 12439 bert_lamb_pretraining true true 4096 2e-3"

cmd_to_run = "bash scripts/run_squad.sh \" --disable_progress_bar \" /results/pretrain_dense/2ndphase-lr0.002/checkpoints/ckpt_1563.ckpt "

cmd_to_run = args.cmd

available_success_trial_cnt = -1
available_test_cnt = -1

while True:
    os.system("nvidia-smi 2>&1 | tee /tmp/nvidia_smi.txt")
    time.sleep(1)
    available_test_cnt += 1
    with open('/tmp/nvidia_smi.txt', 'r') as f:
        lines = f.readlines()
    cnt =  -1
    gpu_found = False
    for line in lines:
        target_line = True
        for k in keywords:
            if k not in line:
                target_line = False
        if target_line is True:
            gpu_found = True
            print("GPU found. Test availability...")
            cnt += 1
            #print("{}:{}".format(cnt,line))
            line_split = line.split()
            #print(line_split)
            pos_mem = line_split.index(keywords[0])
            #print(line_split[pos_mem-2])
            if line_split[pos_mem-2] != '0MiB':
                print(f"GPU {cnt} Memory {line_split[pos_mem-2]} not empty!")
                available_success_trial_cnt = -1
            pos_GPU = line_split.index(keywords[1])
            #print(line_split[pos_GPU-1])
            if line_split[pos_GPU-1] != '0%':
                print(f"GPU {cnt} util {line_split[pos_GPU-1]} not empty!")
                available_success_trial_cnt = -1
    if gpu_found is True:
        available_success_trial_cnt += 1
    print("Available test succesful cnt: {}/5".format(available_success_trial_cnt))
    print("Total available test: {}".format(available_test_cnt))
    print("\nCommand to run when machine are available:")
    print(cmd_to_run)
    if gpu_found and available_success_trial_cnt > 5:
        break
    time.sleep(args.try_sec)

print("Machine is available now")
print(cmd_to_run)
os.system(cmd_to_run)
print("Program ends.")
