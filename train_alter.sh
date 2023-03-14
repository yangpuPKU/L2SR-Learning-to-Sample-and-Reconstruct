python train_alter.py \
    --accelerate 4 \
    --low_frequency_mask_ratio 32 \
    --num_parallel_episodes 16 \
    --num_workers 8 \
    --device 'cuda:4' \
    --random_seed 42 \
    --gamma 1.0 \
    --lr_RL 0.0003 \
    --value_loss_coef 0.5 \
    --entropy_coef 0.001 \
    --update_timestep 28 \
    --num_train_steps 40960000 \
    --num_epochs 10 \
    --lr_recon 0.003 \
    --lr_decay 3 \
    --is_determin 1 \
    --recon_model_checkpoint '../pg_mri/my_reconstructor/half-knee-ssim/acc=4_low=4/best_model.pt' \
    --checkpoints_dir 'alter_checkpoint/acc=4_low=4_knee/lrRL=0.0003_BPdecay=3' \
    --training_dir 'sparse_checkpoint/knee-ssim/acc=4/low=4_ent=0.03_step=40960000/final_model_40960000.zip' \
# python train_alter.py \
#     --dataset 'brain' \
#     --_data_location '../pg_mri/dataset/brain_multicoil' \
#     --sample_rate 0.2 \
#     --center_volume 0 \
#     --resolution 256 \
#     --num_chans 16 \
#     --accelerate 8 \
#     --low_frequency_mask_ratio 32 \
#     --num_parallel_episodes 4 \
#     --num_workers 8 \
#     --device 'cuda:3' \
#     --random_seed 42 \
#     --gamma 1.0 \
#     --lr_RL 0.0001 \
#     --value_loss_coef 0.5 \
#     --entropy_coef 0.03 \
#     --update_timestep 24 \
#     --num_train_steps 40960000 \
#     --num_epochs 10 \
#     --lr_recon 0.003 \
#     --lr_decay 3 \
#     --is_determin 1 \
#     --recon_model_checkpoint '../pg_mri/my_reconstructor/brain-ssim/acc=8_low=8/best_model.pt' \
#     --checkpoints_dir 'Alter_checkpoint/acc=8_low=8_brain/lrRL=0.0001_BPdecay=3' \
#     --training_dir 'parse_checkpoint/brain-ssim/acc=8/low=8_ent=0.03_step=40960000/final_model_40960320.zip' \