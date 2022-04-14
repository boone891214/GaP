
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--list-emb", nargs="+", default=[12, 23])

args = parser.parse_args()
print(args.list_emb)

for i in args.list_emb:
  j = 4*int(i)
  cmd = f"bash scripts/run_pretrain.sh /results/dense_emb{i}-{j} None {i} {j} 20"
  print(cmd)
  os.system(cmd)
