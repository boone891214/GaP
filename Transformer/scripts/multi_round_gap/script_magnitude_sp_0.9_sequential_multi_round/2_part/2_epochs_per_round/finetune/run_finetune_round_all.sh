
EPOCH=40

cd ../../../../..


bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_2_partition_sequential_gap_sp_0.9_ep_2_per_train_no_finetune/finetune_round_10/ '--resume /home/results/multi_round_2_partition_sequential_gap_sp_0.9_ep_2_per_train_no_finetune/10_round/3_prune_2rd_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/2_step_sequential_gap_sp_0.9/finetune.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_2_partition_sequential_gap_sp_0.9_ep_2_per_train_no_finetune/finetune_round_9/ '--resume /home/results/multi_round_2_partition_sequential_gap_sp_0.9_ep_2_per_train_no_finetune/9_round/3_prune_2rd_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/2_step_sequential_gap_sp_0.9/finetune.yaml --restart-training '


