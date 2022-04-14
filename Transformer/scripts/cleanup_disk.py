import os
from time import sleep
import argparse

parser = argparse.ArgumentParser(description='PyTorch seq GaP resnet')
parser.add_argument('--root', type=str, default=None, help='root')
parser.add_argument('--suffix', type=str, default=None, help='root')
args = parser.parse_args()


exclude_list = ["checkpoint_best.pt","checkpoint_last.pt"]

if args.root is None:
    print("Need to specify which dir to clean!")
    exit()

if args.suffix is None:
    print("Need to specify a suffix")
    exit()

print("Root is :", args.root)
print("Suffix is :", args.suffix)

f_list = []
for dirpath, dirs, files in os.walk(args.root):
    for filename in files:
        fname = os.path.join(dirpath,filename)
        if fname.endswith(args.suffix):
            to_exclude = False
            for ex in exclude_list:
                if ex in fname:
                    to_exclude = True
            if to_exclude is True:
                f_list.append(fname)
print("Excluded files:")
for f in f_list:
    print(f)

input("Press any key to continue. File will be removed eternally.")

for dirpath, dirs, files in os.walk(args.root):
    for filename in files:
        fname = os.path.join(dirpath,filename)
        if fname.endswith(args.suffix):
            if fname not in f_list:
                print("To remove:", fname)
                os.system("rm {}".format(fname))
