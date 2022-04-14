
EPOCH=40

cd ../../../../../..


bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_1_partition_sequential_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_10/ '--resume /home/results/multi_round_1_partition_sequential_gap_sp_0.9_ep_10_per_train_no_finetune/10_round/2_prune_all_grow_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/1_step_sequential_gap_sp_0.9/all_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_1_partition_sequential_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_5/ '--resume /home/results/multi_round_1_partition_sequential_gap_sp_0.9_ep_10_per_train_no_finetune/5_round/2_prune_all_grow_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/1_step_sequential_gap_sp_0.9/all_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_1_partition_sequential_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_1/ '--resume /home/results/multi_round_1_partition_sequential_gap_sp_0.9_ep_10_per_train_no_finetune/1_round/2_prune_all_grow_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/1_step_sequential_gap_sp_0.9/all_0.9_masked.yaml --restart-training '

