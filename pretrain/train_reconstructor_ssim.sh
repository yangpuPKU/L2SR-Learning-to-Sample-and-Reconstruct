CUDA_VISIBLE_DEVICES=0 python -m src.train_reconstruction_ssim \
    --data_path "dataset/knee_singlecoil" \
    --accelerations 4 \
    --center_fractions 0.125 \
    --sample_rate 0.5 \
    --num_chans 16 \
    --exp_dir 'my_reconstructor/knee-ssim/acc=4_center=8'
# CUDA_VISIBLE_DEVICES=1 python -m src.train_reconstruction_ssim \
#     --dataset brain \
#     --data_path "dataset/brain_multicoil" \
#     --accelerations 8 \
#     --center_fractions 0.03125 \
#     --resolution 256 \
#     --center_volume False \
#     --sample_rate 0.2 \
#     --exp_dir 'my_reconstructor/brain-ssim/acc=8_center=8'