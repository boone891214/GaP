
EPOCH=40

cd ../../../../..

bash scripts/run_DGX1_AMP_8GPU.sh 'amp' 1 0.000846 4000 ${EPOCH} 5120 8 /results/multi_round_6_partition_sequential_gap_sp_0.8_ep_10_per_train_no_finetune/finetune_round_2/ '--resume /home/results/multi_round_6_partition_sequential_gap_sp_0.8_ep_10_per_train_no_finetune/2_round/7_prune_6th_grow_1st_blk/checkpoints/checkpoint_last.pt --sp-retrain --sp-prune-before-retrain --sp-admm-sparsity-type irregular --sp-config-file /home/profiles/6_step_sequential_gap_sp_0.8/finetune.yaml --restart-training '




