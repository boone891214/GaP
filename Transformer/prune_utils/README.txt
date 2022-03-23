# DNN prunning algorithm.

# Function explanation
prune_main.py
    |
    | --> prune_init(args, model) # initialize the prune function
    |
    | --> prune_update(epoch=0, batch_idx=0) # update ADMM variables (U, Z) during training
    |
    | --> prune_update_combined_loss(loss) # update the loss in ADMM training
    |
    | --> prune_harden() # A hard thresholding to set smallest values to 0 (this is done after the last epoch in training)
    |
    | --> prune_apply_masks() # Apply a binary mask on the weights (This is done in retraining to force unwanted weights to be 0)
    | 
    | --> prune_store_weights() # store the weights into files
    |
    | --> prune_print_sparsity(model) # print the sparsity of a model
    |
    | --> prune_update_learning_rate(optimizer, epoch, args) # update the learning rate of ADMM trainer. ADMM can use a different LR schedular because each time prune_update() is called, the ADMM is trying to solve a new problem.

  # Dependencies

  prune_main.py
    |
    | --> prune_base.py (Base class of prune) --> admm.py (ADMM pruning algorithm) --> multi_level_admm.py (Pruning while fixing part of the model)
    |           |
    |           | --> L_1_reweighted.py (L1 reweighted pruning algorithm (TO DO))
    |
    | --> retrain.py (retrain/fine-tune the network after hard prune) --> multi_level_retrain.py (retrain while fixing part of the model)


# Examples:
from prune_utils import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',help='path to dataset')
prune_parse_arguments(parser)
args = parser.parse_args()

def main():
    # define dataloader
    trainDataLoader = torch.utils.data.DataLoader(...)
    testDataLoader = torch.utils.data.DataLoader(...)

    # define DNN model
    model = ... 
    model = torch.nn.DataParallel(model).cuda() # if using multiple GPU

    # define loss function
    criterion = ...


    # define optimizer
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

    # ADMM PRUNE
    prune_init(args, model)

    prune_apply_masks() # if wanted to make sure the mask is applied in retrain

    prune_print_sparsity(model) # check sparsity before retrain

    for epoch in range(start_epoch,args.epoch):
        
        # ADMM PRUNE
        prune_update(epoch)

        scheduler.step()

        # ADMM PRUNE
        prune_update_learning_rate(optimizer, epoch, args)

        for batch_id, data in enumerate(trainDataLoader):
            model.train()

            output = model(input)

            loss = criterion(...) # regular loss, i.e., cross-entropy, mse, ... 

            # ADMM PRUNE
            loss = prune_update_loss(loss)

            loss.backward()
            optimizer.step()

            # ADMM PRUNE
            prune_apply_masks()

        if epoch == args.epoch - 1:
            # ADMM PRUNE
            prune_harden()

        # save the model
        save_path = 'path_name.pth'
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)


# Training scripts

# regular train
python3 imagenet_main.py <data_folder> --arch=resnet50 --worker=16 --batch-size=256 --gpu_id=1 --epochs=180 --learning-rate=0.1 --resume=existing_ckpt.pth

# admm
python3 imagenet_main.py <data_folder> --arch=resnet50 --worker=16 --batch-size=256 --gpu_id=1 --epochs=120 --learning-rate=0.1 --resume=pretrained.ckpt --sp-admm --sp-config-file=./profile/config_resnet50.yaml --sp-admm-update-epoch=30 --sp-admm-sparsity-type=irregular --sp-admm-lr=0.01

# retrain
python3 imagenet_main.py <data_folder> --arch=resnet50 --worker=16 --batch-size=256 --gpu_id=1 --epochs=120 --learning-rate=0.001 --resume=hard_pruned.ckpt --sp-retrain  --sp-config-file=./profile/config_resnet50.yaml

# evaluate
python3 imagenet_main.py <data_folder> --arch=resnet50 --worker=16 --batch-size=256 --gpu_id=1 --evaluate --resume=existing_ckpt.pth

# 2:4 structured pruning using admm
python3 imagenet_main.py <data_folder> --arch=resnet50 --worker=16 --batch-size=256 --gpu_id=1 --epochs=120 --learning-rate=0.1 --resume=pretrained.ckpt --sp-admm --sp-config-file=./profile/config_resnet50.yaml --sp-admm-update-epoch=30 --sp-admm-lr=0.01 --sp-admm-sparsity-type=N:M-prune-pattern --sp-admm-select-number 2 --sp-admm-pattern-row-sub 1 --sp-admm-pattern-col-sub 4 


