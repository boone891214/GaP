# RigL-PyTorch

An open source implementation of Google Research's paper: [Rigging the Lottery: Making All Tickets Winners](https://proceedings.mlr.press/v119/evci20a/evci20a.pdf) (RigL) in PyTorch with the [NVIDIA deep learning example codebase](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets). 

This codebase is also used to reproduce the RigL results in the ICLR 2022 paper "[Effective Model Sparsification by Scheduled Grow-and-Prune Methods](https://openreview.net/pdf?id=xa6otUDdP2W)".


## Requirements

For easy implementation, we suggest to use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) with CUDA-11 for the training environments.
We have pre-built the ready-to-run nvidia-docker image [here](https://drive.google.com/file/d/1kEXD8ZHXEoHIMSpAFKaKZh2SExoWHONy/view?usp=sharing).

- Load pre-built docker images (download or build): 
  
    `docker load -i nvidia_rn50.tar`


- Rename the docker image: 
  
    `docker image tag 4c5875fdd48859f69015c7ec7183e5d2e706ffe7dabcad177e39e041673dba82 nvidia_rn50:latest`


- Start nvidia-docker interactive session: 
  
    `nvidia-docker run --rm -it -v /path/to/your/imagenet/:/data/imagenet -v /path/to/your/project:/workspace/rn50 --ipc=host nvidia_rn50`



## ImageNet-1k results

### `Sparsity ratio 0.8`

| Models  | Method | Epoch | Sparsity Ratio | Sparsity Distribution | Top-1 Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ResNet-50 | RigL | 100 | 0.8 | Uniform | 74.6% |
| ResNet-50 | RigL | 100 | 0.8 | ERK | 75.4% |
| ResNet-50 | RigL | 500 | 0.8 | Uniform | 76.9% |
| ResNet-50 | RigL | 500 | 0.8 | ERK | 77.4% |
| ResNet-50 | RigL | 1200 | 0.8 | Uniform | 77.1% |
| ResNet-50 | RigL | 1200 | 0.8 | ERK | 77.4% |

### `Sparsity ratio 0.9`

| Models  | Method | Epoch | Sparsity Ratio | Sparsity Distribution | Top-1 Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ResNet-50 | RigL | 100 | 0.9 | Uniform | 72.5% |
| ResNet-50 | RigL | 100 | 0.9 | ERK | 73.9% |
| ResNet-50 | RigL | 500 | 0.9 | Uniform | 75.6% |
| ResNet-50 | RigL | 500 | 0.9 | ERK | 76.3% |
| ResNet-50 | RigL | 1200 | 0.9 | Uniform | 76.0% |
| ResNet-50 | RigL | 1200 | 0.9 | ERK | 76.8% |

### References
```
@inproceedings{evci2020rigging,
  title={Rigging the lottery: Making all tickets winners},
  author={Evci, Utku and Gale, Trevor and Menick, Jacob and Castro, Pablo Samuel and Elsen, Erich},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={2943--2952},
  year={2020},
  organization={PMLR}
}

@inproceedings{ma2022effective,
    title={Effective Model Sparsification by Scheduled Grow-and-Prune Methods},
    author={Xiaolong Ma and Minghai Qin and Fei Sun and Zejiang Hou and Kun Yuan and Yi Xu and Yanzhi Wang and Yen-Kuang Chen and Rong Jin and Yuan Xie},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2022},
    url={https://openreview.net/forum?id=xa6otUDdP2W}
}
```


