
EPOCH=40

cd ../../../../../..

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_10/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/10_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_9/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/9_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_8/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/8_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_7/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/7_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_6/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/6_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_5/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/5_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_4/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/4_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_3/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/3_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_2/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/2_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/finetune_round_1/ '--resume /home/results/multi_round_3_partition_parallel_gap_sp_0.9_ep_10_per_train_no_finetune/1_round/4_combine_all_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/3_step_parallel_gap_sp_0.9/all_layer_0.9_masked.yaml --restart-training '



