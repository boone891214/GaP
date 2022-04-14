import os
from time import sleep
import argparse

parser = argparse.ArgumentParser(description='PyTorch seq GaP Transformer')
parser.add_argument('--ep-per-step', type=int, default=5, help='epoch per step')
parser.add_argument('--num-rounds', type=int, default=3, help='total number of round')
parser.add_argument('--extra-cmd', type=str, default="", help='extra cmd, such as --short-train or --quick-save')
parser.add_argument('--sparsity-type', type=str, default="4:2-H-V-balanced", help='sparsity type')
parser.add_argument('--global-workspace', type=str, default=None, help='sparsity type')
parser.add_argument('--sparsity', type=float, default=0.8, help='sparsity')
parser.add_argument('--seed', type=int, default=3, help='seed for train')

args = parser.parse_args()

# synchronize files to /workspace/translation/
os.system("cp /home/*.py /workspace/translation/")
os.system("cp /home/*.sh /workspace/translation/")
os.system("cp -r /home/scripts/* /workspace/translation/scripts/")
os.system("mkdir -p /workspace/translation/prune_utils/")
os.system("cp /home/prune_utils/*.py /workspace/translation/prune_utils/")
os.system("cp -r /home/fairseq/modules/*.py  /workspace/translation/fairseq/modules/")
os.system("cp /home/fairseq/*.py  /workspace/translation/fairseq/")

epoch_per_train=args.ep_per_step
extra_cmd = args.extra_cmd  #"--short-train --quick-save" # --short-train  --no-checkpoints
sparsity = args.sparsity
sparsity_type = args.sparsity_type
seed = args.seed

if sparsity_type == "4:2-H-V-balanced":
    folder_name = "4-2-HVB"
elif sparsity_type == "4:2-2:1":
    folder_name = "4-2-2-1"
else:
    folder_name = sparsity_type

if args.global_workspace is None:
    global_workspace = "/results/GaP/3_step_forward_gap/{0}/multi_round_3_partition_sequential_gap_ep_{1}_per_train/".format(folder_name, epoch_per_train)
else:
    global_workspace = args.global_workspace

os.system("mkdir -p {}".format(global_workspace))
local_workspace = ['None'] #[global_workspace +  "/round_1/step_1/"]
log_file=global_workspace + "log.txt"
os.system("rm -f {}".format(log_file))
os.system("touch {}".format(log_file))
extra_extra_cmd = " 2>&1 | tee -a {}".format(log_file)

cmd = []

config_file = "/home/profiles/3_step_forward_gap/{0}_std_naming/step_0.yaml".format(sparsity)
local_workspace.append(global_workspace +  "/round_0/step_0/")
resume = local_workspace[-2]; # + "/checkpoints/checkpoint_best.pt".format(epoch_per_train-1)

cmd.append("mkdir -p {1} ; bash scripts/run_DGX1_AMP_8GPU.sh 'amp' {7} 0.000846 4000 {0} 5120 8 {1} \"--restart-training --resume {2} --sp-retrain --sp-prune-before-retrain  --sp-admm-sparsity-type {3} --sp-config-file {4} {5}\" {6}".format(epoch_per_train, local_workspace[-1], resume, sparsity_type, config_file, extra_cmd, extra_extra_cmd, seed) )
print(cmd[-1])
os.system(cmd[-1])


for round in range(1,args.num_rounds+1):

    for step in [1,2,3]:
        os.system("nvidia-smi")
        local_workspace.append(global_workspace +  "/round_{}/step_{}/".format(round, step))

        config_file = "/home/profiles/3_step_forward_gap/{0}_std_naming/step_{1}.yaml".format(sparsity, step)

        resume = local_workspace[-2] + "/checkpoints/checkpoint{}.pt".format(epoch_per_train)

        cmd.append("mkdir -p {1} ; bash scripts/run_DGX1_AMP_8GPU.sh 'amp' {7} 0.000846 4000 {0} 5120 8 {1} \"--restart-training --resume {2} --sp-retrain --sp-prune-before-retrain  --sp-admm-sparsity-type {3} --sp-config-file {4} {5}\" {6}".format(epoch_per_train, local_workspace[-1], resume, sparsity_type, config_file, extra_cmd, extra_extra_cmd, seed) )
        print(cmd[-1])
        os.system(cmd[-1])

    sleep(10)
