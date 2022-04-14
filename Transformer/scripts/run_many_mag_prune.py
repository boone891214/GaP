
import os
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("--list-emb", nargs="+", default=[12, 23])


r_c_s_s = [[32,8,0.8,0.99],[32,8,0.8,1.0],
           [32,8,0.9,0.99],[32,8,0.9,1.0],
           [32,1,0.8,0.99],[32,1,0.8,1.0],
           [32,1,0.9,0.99],[32,1,0.9,1.0],
           [8,1,0.8,0.99],[8,1,0.8,1.0],
           [8,1,0.9,0.99],[8,1,0.9,1.0],
           [8,8,0.8,0.99],[8,8,0.8,1.0],
           [8,8,0.9,0.99],[8,8,0.9,1.0],
           [16,8,0.8,0.99],[16,8,0.8,1.0],
           [16,8,0.9,0.99],[16,8,0.9,1.0],]

cmd = []
for config in r_c_s_s:
    cmd.append("bash scripts/run_mag_prune.sh /results/mag_1_shot/block_{0}x{1}_{2}_and_irregular_{3}/  results/dense/ep40/checkpoint_best.pt 0.000846 40 block_and_irregular profiles/all_layer_0.8.yaml \"--sp-block-irregular-sparsity ({2},{3}) --sp-admm-block ({0},{1})  --no-epoch-checkpoints\"".format(config[0], config[1], config[2], config[3]))



for c in cmd:
    print(c)
    os.system("nvidia-smi")
    os.system(c)
