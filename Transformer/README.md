# PyTorch Implementation of Transformer Grow and Prune
ICLR 2022 "[Effective Model Sparsification by Scheduled Grow-and-Prune Methods](https://openreview.net/pdf?id=xa6otUDdP2W)".

This Readme explains how to run the scheduled grow-and-prune for sparse Transformers.



## Setup

The following section lists the requirements in order to start training the Transformer model.



## Quick Start Guide
To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the Transformer model on the [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository 
```
git clone https://github.com/jybbjybb/Transformer_GaP.git
cd Transformer_GaP
```

2. Build Transformer PyTorch NGC  container
```bash
docker build . -t your.repository:transformer
```

If you already have preprocessed data, go to the next step.

Otherwise, follow https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer to download the data.

Download and process the data by go into the docker image
```
nvidia-docker run -it --rm --ipc=host your.repository:transformer bash
```
If you already have data downloaded, but it has not yet been preprocessed, use:
```bash
nvidia-docker run -it --rm --ipc=host -v <path to your unprocessed data>:/workspace/translation/examples/translation/orig your.repository:transformer bash
```
Download and preprocess dataset: Download and preprocess the WMT14 English-German dataset.

```bash 
scripts/run_preprocessing.sh
```
After running this command, data will be downloaded to `/workspace/translation/examples/translation/orig` directory and this data will be processed and put into `/data/wmt14_en_de_joined_dict` directory. 

3. Restart the docker 
```
bash scripts/start_docker.sh 
```

4. Start the GaP
```
python scripts/run_seq_gap.py --ep-per-step 2 --num-steps 3 --extra-cmd='--no-epoch-checkpoints' --extra-cmd-step0='--no-epoch-checkpoints' --global-workspace results/tmp/ --sparsity 0.8 --config-folder profiles/3_step_forward_gap/0.8_std_naming/ --num-parts 3 --partition-type cyclic
```

5. Test the results
```
bash scripts/run_test.sh <your saved checkpoints>
```

# Argument explanation
```
--ep-per-step: type=int, epochs to train per grow-and-prune step
--num-steps: type=int, number of steps of the GaP
--extra-cmd: type=str, extra cmd, such as --no-epoch-checkpoints to append to the training cmd.
--extra-cmd-step0: type=str, extra cmd, such as --no-epoch-checkpoints to append to the first step of training cmd.
--sparsity-type: type=str, sparsity type, use "irregular" throughout this paper.
--global-workspace:, type=str, working directory
--sparsity: type=float, sparsity when use global sparse distribution (not used if sparsity yaml files are assigned)
--config-folder: type=str, this folder contains yaml file of GaP in this folder, named as step_0.yaml, step_1.yaml, ...
--num-parts: type=int, number of partitions
--partition-type: type=str, cyclic or random partition. Use cyclic for better results
--precision:, type=str, "amp" to turn on mixed fp16, use " " to use fp32
```

