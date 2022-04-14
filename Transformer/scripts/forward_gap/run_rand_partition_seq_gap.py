import os
from time import sleep
import argparse

parser = argparse.ArgumentParser(description='PyTorch seq GaP Transformer')
parser.add_argument('--ep-per-step', type=int, default=5, help='epoch per step')
parser.add_argument('--seed', type=int, default=3, help='seed for train')
parser.add_argument('--num-steps', type=int, default=100, help='number of steps')
parser.add_argument('--extra-cmd', type=str, default="", help='extra cmd, such as --short-train or --quick-save --sp-global-weight-sparsity or --sp-prune-threshold 0.3')
parser.add_argument('--extra-cmd-step0', type=str, default="", help='extra cmd, for step 0')
parser.add_argument('--global-workspace', type=str, default=None, help='sparsity type')
parser.add_argument('--sparsity', type=float, default=0.8, help='sparsity')
parser.add_argument('--warmup', type=int, default=8, help='warmup')
parser.add_argument('--resume', type=str, default='None', help='resume from a checkpoint to start gap')
parser.add_argument('--config-folder', type=str, default='None', help='yaml file of gap in this folder')
parser.add_argument('--bs', type=int, default=256, help='bs per gpu')
parser.add_argument('--optimizer-batch-size', type=int, default=2048, help='bs per gpu')
parser.add_argument('--num-parts', type=int, default=3, help='number of partitions')
parser.add_argument('--lr', type=float, default=0.000846, help='learning rate')
parser.add_argument('--lr-schedule-within-step', type=str, default='not defined', help='lr scheduler, inner loop, each training')
parser.add_argument('--lr-schedule-across-step', type=str, default='constant', help='lr scheduler, outer loop')
parser.add_argument('--lr-schedule-across-step-decay-factor', type=float, default=1.0, help='higher level lr decay factor')
args = parser.parse_args()
# synchronize files to /workspace/translation/
os.system("cp /home/*.py /workspace/translation/")
os.system("cp /home/*.sh /workspace/translation/")
os.system("cp -r /home/scripts/* /workspace/translation/scripts/")
os.system("mkdir -p /workspace/translation/prune_utils/")
os.system("cp /home/prune_utils/*.py /workspace/translation/prune_utils/")
os.system("cp -r /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/")
os.system("cp /home/fairseq/*.py  /workspace/translation/fairseq/")

def lr_cosine_policy(base_lr, warmup_length, epoch, total_epochs, logger=None):
    if epoch < warmup_length:
        lr = base_lr * (epoch + 1) / warmup_length
    else:
        e = epoch - warmup_length
        es = total_epochs - warmup_length
        lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
    return lr

def lr_linear_policy(base_lr, warmup_length, epoch, total_epochs, logger=None):
    if epoch < warmup_length:
        lr = base_lr * (epoch + 1) / warmup_length
    else:
        e = epoch - warmup_length
        es = total_epochs - warmup_length
        lr = base_lr * (1 - (e / es))
    return lr

def lr_const_policy(base_lr, warmup_length, epoch, total_epochs=0, logger=None):
    if epoch < warmup_length:
        lr = base_lr * (epoch + 1) / warmup_length
    else:
        lr = base_lr
    return lr

def lr_step_policy(base_lr, warmup_length, epoch, step, decay_factor=0.99, logger=None):
    if epoch < warmup_length:
        lr = base_lr * (epoch + 1) / warmup_length
    else:
        lr = base_lr
        decay_times = int(epochs/step)
        for i in range(decay_times):
            lr *= decay_factor
    return lr


epoch_per_train=args.ep_per_step
#extra_cmd = args.extra_cmd  #"--short-train --quick-save" # --short-train  --no-checkpoints

seed = args.seed

num_parts = args.num_parts
num_steps = args.num_steps

sparsity = args.sparsity
sparsity_type = args.sparsity_type

resume = args.resume
warmup = args.warmup
bs = args.bs
optimizer_batch_size = args.optimizer_batch_size
config_folder = args.config_folder
base_lr = args.lr
lr = base_lr
lr_sch_within_step = args.lr_schedule_within_step
lr_sch_across_step = args.lr_schedule_across_step

if sparsity_type == "4:2-H-V-balanced":
    folder_name = "4-2-HVB"
elif sparsity_type == "N:M-prune-pattern":
    folder_name = "N-M-no-balanced"
else:
    folder_name = sparsity_type

if args.global_workspace is None:
    global_workspace = f"/results/GaP/rand_{num_parts}_partition_seq_gap/{folder_name}/multi_round_3_partition_sequential_gap_ep_{epoch_per_train}_per_train/"
else:
    global_workspace = args.global_workspace

os.system("mkdir -p {}".format(global_workspace))
local_workspace = ['None'] #[global_workspace +  "/round_1/step_1/"]
log_file=global_workspace + "log.txt"
os.system("rm -f {}".format(log_file))
os.system("touch {}".format(log_file))
extra_extra_cmd = " 2>&1 | tee -a {}".format(log_file)

cmd = []

for step in range(num_steps):

    os.system("nvidia-smi")

    if lr_sch_across_step == 'constant':
        lr = lr_const_policy(base_lr, 0, step, num_steps)
    elif lr_sch_across_step == 'cosine':
        lr = lr_cosine_policy(base_lr, 0, step, num_steps)
    elif lr_sch_across_step == 'linear':
        lr = lr_linear_policy(base_lr, 0, step, num_steps)
    elif lr_sch_across_step == 'step':
        lr = lr_step_policy(base_lr, 0, step, num_steps, 1, args.lr_schedule_across_step_decay_factor)
    else:
        print("Higher level LR not defined in [constant,cosine,linear,step]")
        input("?")

    if step == 0:
        extra_cmd = args.extra_cmd_step0
    else:
        extra_cmd = args.extra_cmd

    local_workspace.append(global_workspace +  f"/step_{step}/")

    extra_cmd += f" 2>&1 | tee {local_workspace[-1]}/log.txt"

    config_file = f"/home/profiles/rand_{num_parts}_seq_gap_sp_{sparsity}/step_{step}.yaml"

    resume = local_workspace[-2] + "/checkpoints/checkpoint_last.pt"

    os.system(f"mkdir -p {local_workspace[-1]}")

    cmd.append(f"bash scripts/run_DGX1_AMP_8GPU.sh 'amp' {seed} {lr} 4000 {epoch_per_train} 5120 8 {local_workspace[-1]} \"--restart-training --resume {resume} --sp-retrain --sp-prune-before-retrain  --sp-admm-sparsity-type {sparsity_type} --sp-config-file {config_file} {extra_cmd}\" {extra_extra_cmd}")
    print(cmd[-1])
    os.system(cmd[-1])

    sleep(10)
