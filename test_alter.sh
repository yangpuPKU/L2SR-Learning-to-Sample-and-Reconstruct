python test_alter.py \
    --accelerate 4 \
    --low_frequency_mask_ratio 32 \
    --num_parallel_episodes 64 \
    --device 'cuda:0' \
    --random_seed 42 \
    --recon_model_checkpoint '' \
    --training_dir '' \
# python test_sparse.py \
#     --dataset 'brain' \
#     --_data_location '../pg_mri/dataset/brain_multicoil' \
#     --sample_rate 0.2 \
#     --center_volume 0 \
#     --resolution 256 \
#     --accelerate 16 \
#     --low_frequency_mask_ratio 32 \
#     --num_parallel_episodes 64 \
#     --device 'cuda:0' \
#     --random_seed 42 \
#     --num_chans 8 \
#     --recon_model_checkpoint '' \
#     --training_dir '' \