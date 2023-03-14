python test_sparse.py \
    --dataset 'knee' \
    --_data_location '../pg_mri/dataset/knee_singlecoil' \
    --accelerate 4 \
    --low_frequency_mask_ratio 32 \
    --num_parallel_episodes 64 \
    --device 'cuda:0' \
    --random_seed 42 \
    --num_chans 16 \
    --recon_model_checkpoint '../pg_mri/my_reconstructor/half-knee-ssim/acc=4_low=4/best_model.pt' \
    --training_dir 'sparse_checkpoint/knee-ssim/acc=4/low=4_ent=0.03_step=40960000/final_model_40960000.zip' \
# python test_sparse.py \
#     --dataset 'brain' \
#     --_data_location '../pg_mri/dataset/brain_multicoil' \
#     --sample_rate 0.2 \
#     --center_volume 0 \
#     --resolution 256 \
#     --accelerate 8 \
#     --low_frequency_mask_ratio 64 \
#     --num_parallel_episodes 64 \
#     --device 'cuda:1' \
#     --random_seed 42 \
#     --num_chans 16 \
#     --recon_model_checkpoint '../pg_mri/my_reconstructor/brain-ssim/acc=8_low=8/best_model.pt' \
#     --training_dir 'sparse_checkpoint/brain-ssim/acc=8/low=8_ent=0.03_step=40960000/final_model_40960000.zip' \