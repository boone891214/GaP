import os

import numpy as np
import csv

import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--dir", default=None, type=str, help="dir to find ckpt")
args = parser.parse_args()

calculate_mask = False
calculate_mask_change = True

if calculate_mask:
    import torch
    #dir = "checkpoints/gap/irregular/0.9_conv1_0.9_fc_0.9/multi_round_cyclic_4_part_seq_gap_ep_30_per_train_global_wt/"
    #dir = "checkpoints/gap/irregular/0.9_conv1_0.9_fc_0.9/multi_round_3_rand_part_seq_gap_ep_20_per_train/"
    dir = args.dir
    #"/nasmnt/xiaolong.ma/nv_Transformer/results/multi_round_3_partition_sequential_gap_sp_0.9_ep_10_per_train_no_finetune/"
    ckpt_list = []

    for subdir, dirs, files in sorted(os.walk(dir)):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            #print(filepath)
            if filepath.endswith("checkpoint_best.pt") and "finetune_round" not in filepath:
                print (filepath)
                ckpt_list.append(filepath)

    def sorting_func(lst):
      return len(lst)

    ckpt_list = sorted(ckpt_list, key=sorting_func)
    print("CKPTs are:")
    for cc in ckpt_list:
        print(cc)
    ckpt_list = ckpt_list[:]


    masks = []
    densities = []
    for ckpt in ckpt_list:

        ck = torch.load(ckpt)

        print(f"{ckpt} loaded!")
        mm = {}
        den = {}
        for key in ck['model']:

            if 'weight' in key and 'norm' not in key and "ln" not in key and "embed" not in key:
                #print(key,ck['state_dict'][key].shape)
                #print(key, ck['model'][key].shape)
                mm[key] = ck['model'][key].detach().cpu().numpy() != 0
                den[key] = float(np.sum(mm[key]))/np.size(mm[key])
        masks.append(mm)
        densities.append(den)
        #print(den)
        del ck
        print(len(mm))
    #exit()

    IoU = {}
    dd = {}
    for key in masks[0]: #key = layer names
        IoU[key] = np.zeros([len(masks),len(masks)])

    for key in densities[0]:
        dd[key] = np.zeros(len(masks))
        for i in range(len(densities)):
            dd[key][i] = densities[i][key]


    for i in range(len(masks)):
        for j in range(i, len(masks)):
            print("Processing:",i,j)
            for key in masks[i]:
                I = masks[i][key] * masks[j][key] != 0
                U = (masks[i][key] + masks[j][key]) > 0
                #print(float(np.sum(I))/np.size(I))
                if np.sum(U) > 0:
                    iou = float(np.sum(I))/float(np.sum(U))
                else:
                    iou = 0
                IoU[key][i][j] = iou
                #print(i,j, key, iou)


    os.system(f"mkdir -p {dir}/all_iou/")
    for key in IoU:
        file = f'{dir}/all_iou/{key}.csv'
        os.system(f"rm -rf {file}")

        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(dd[key])
            writer.writerow("masks:")
            for rr in IoU[key]:
                writer.writerow(rr)

if calculate_mask_change:
    #filename = "all_iou/all_iou_rand_3_gl_wt/module.layer2.1.conv3.weight.csv"
    dir = "all_iou/all_iou_cyclic_3_unif_0.9/"
    all_files = []
    m_d = {}
    for subdir, dirs, files in sorted(os.walk(dir)):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".csv"):
                #print (filepath)
                all_files.append(filepath)
    for filename in all_files:
        masks = {}
        dd = {}
        masks[filename] = []
        with open(filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            i = 0
            for row in spamreader:
                #print(row)
                #print(row[0].split(","))
                if i == 0:
                    dd[filename] = [float(ele) for ele in row[0].split(",")]
                    #print(res)
                    #input("?")
                elif i > 1:
                    res = [float(ele) for ele in row[0].split(",")]
                    #print(res)
                    masks[filename].append(res)

                i += 1
        masks[filename] = np.array(masks[filename])
        #print(masks[filename])
        #print(dd[filename])
        #print(masks[filename].shape)
        den = dd[filename]
        #print(den)
        mask_diff = []
        for i in range(len(den)-2):
            if den[i+1] - den[i] > 0.3 and den[i+1] - den[i+2] > 0.3: # a grow and prune event
                mask_diff.append(masks[filename][i][i+2])
        #print(filename)
        #print(mask_diff)
        m_d[filename] = mask_diff
    for key in sorted(m_d):
        print(key, len(m_d[key]))
    #print(m_d)
    os.system("rm -rf ./mask_update_diff.csv")
    for key in sorted(m_d):
        with open(f'./mask_update_diff.csv', 'a') as f:
            writer = csv.writer(f)
            write_list = m_d[key]
            write_list.insert(0,key.split("/")[-1])
            print(write_list)
            writer.writerow(write_list)
