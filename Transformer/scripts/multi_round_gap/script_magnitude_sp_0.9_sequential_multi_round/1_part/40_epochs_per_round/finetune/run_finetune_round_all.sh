
EPOCH=40

cd ../../../../../..


bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_1_partition_sequential_gap_sp_0.9_ep_40_per_train_with_finetune/finetune_round_5/ '--resume /home/results/multi_round_1_partition_sequential_gap_sp_0.9_ep_40_per_train_with_finetune/5_round/2_prune_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/1_step_sequential_gap_sp_0.9/all_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_1_partition_sequential_gap_sp_0.9_ep_40_per_train_with_finetune/finetune_round_4/ '--resume /home/results/multi_round_1_partition_sequential_gap_sp_0.9_ep_40_per_train_with_finetune/4_round/2_prune_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/1_step_sequential_gap_sp_0.9/all_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_1_partition_sequential_gap_sp_0.9_ep_40_per_train_with_finetune/finetune_round_3/ '--resume /home/results/multi_round_1_partition_sequential_gap_sp_0.9_ep_40_per_train_with_finetune/3_round/2_prune_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/1_step_sequential_gap_sp_0.9/all_0.9_masked.yaml --restart-training '

