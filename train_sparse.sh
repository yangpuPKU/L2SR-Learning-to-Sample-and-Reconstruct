python train_sparse.py \
    --accelerate 4 \
    --low_frequency_mask_ratio 32 \
    --num_parallel_episodes 16 \
    --num_workers 8 \
    --device 'cuda:4' \
    --random_seed 42 \
    --gamma 1.0 \
    --lr 0.0003 \
    --value_loss_coef 0.5 \
    --entropy_coef 0.03 \
    --recon_model_checkpoint '../pg_mri/my_reconstructor/half-knee-ssim/acc=4_low=4/best_model.pt' \
    --checkpoints_dir 'sparse_checkpoint/knee-ssim/acc=4/low=4_ent=0.03_step=40960000' \
    --update_timestep 28 \
    --num_train_steps 4096000 \
# python train_sparse.py \
#     --dataset 'brain' \
#     --_data_location '../pg_mri/dataset/brain_multicoil' \
#     --sample_rate 0.2 \
#     --center_volume 0 \
#     --resolution 256 \
#     --accelerate 8 \
#     --low_frequency_mask_ratio 32 \
#     --num_parallel_episodes 4 \
#     --num_workers 8 \
#     --device 'cuda:4' \
#     --random_seed 42 \
#     --gamma 1.0 \
#     --lr 0.0003 \
#     --value_loss_coef 0.5 \
#     --entropy_coef 0.03 \
#     --num_chans 16 \
#     --recon_model_checkpoint '../pg_mri/my_reconstructor/brain-ssim/acc=8_low=8/best_model.pt' \
#     --checkpoints_dir 'sparse_checkpoint/brain-ssim/acc=8/low=8_ent=0.03_step=40960000' \
#     --update_timestep 24 \
#     --num_train_steps 4096000 \