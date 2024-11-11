import argparse
import json
import random
from matplotlib.style import use
import numpy as np
import torch
import os
from activemri.envs.sparse_vecenvs import SparseVecEnv
from activemri.feature_extractor import extractor
from stable_baselines3.common import env_checker
from stable_baselines3 import A2C
import matplotlib
import time
matplotlib.use('Agg')

def train(args):
    #####################
    #---- build env ----#
    #####################
    train_env = SparseVecEnv(args, mode='train')
    print('--- Successfully load environment ---\n')
    print("Number of available actions:", train_env.action_space.n)
    # env_checker.check_env(env)


    ##################
    #---- policy ----#
    ##################
    policy_kwargs = {
        'net_arch': dict(), 
        'features_extractor_class': extractor.Extractor, 
        'features_extractor_kwargs': {'opts': args}, 
    }
    
    #################
    #---- train ----#
    #################
    model = A2C(
        policy = 'MultiInputPolicy', 
        env = train_env, 
        learning_rate = args.lr, 
        n_steps = args.update_timestep, 
        gamma = args.gamma, 
        use_rms_prop = args.use_rms_prop, 
        gae_lambda = args.gae_lambda, 
        ent_coef = args.entropy_coef, 
        vf_coef = args.value_loss_coef, 
        tensorboard_log = args.checkpoints_dir,
        policy_kwargs = policy_kwargs, 
        device = args.device, 
        verbose = 1, 
        seed = args.random_seed, 
    )
    model.policy.action_net = extractor.Action_net().to(model.device)  # TODO: Fix the bug that the model.policy.optimizer doesn not contain such customized action_net!
    print(model.policy)
    print('n_envs:', model.n_envs)
    # resume
    if args.training_dir:
        model.set_parameters(args.training_dir, device=model.device)

    # learn
    for i in range(10):
        model.learn(total_timesteps=args.num_train_steps, log_interval=100, reset_num_timesteps=(i==0))
        model.save(args.checkpoints_dir+f'/final_model_{model.num_timesteps}')
    
    
def set_random_seeds(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    

def build_args():
    parser = argparse.ArgumentParser(description='MRI Reconstruction Example')

    # env
    parser.add_argument("--env_type", type=str, default='sparse')
    
    # MRI setting parameters
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--accelerate", type=int, default=4)
    parser.add_argument("--num_parallel_episodes", type=int, default=4)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--training_dir", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument(
        "--reward_metric",
        type=str,
        choices=["mse", "ssim", "nmse", "psnr"],
        default="ssim",
    )
    # parser.add_argument("--resume", action="store_true")
    
    # A2C training parameters
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--use_rms_prop", type=int, choices=[0, 1], default=1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--update_timestep", type=int, default=1000)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # model parameters
    parser.add_argument('--fc_size', default=256, type=int, help='Size (width) of fully connected layer(s).')
    parser.add_argument("--ppo_model_type",type=str,default="pg_mri")
    
    # Mask parameters
    parser.add_argument('--accelerations', nargs='+', default=[8], type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for '
                             'each volume.')
    parser.add_argument("--low_frequency_mask_ratio", type=int, default=8)
    parser.add_argument("--apply_attrs_padding", type=int, default=0, choices=[0, 1])
    
    # Reconstructor parameters
    parser.add_argument("--recon_model_checkpoint", type=str, default='../pg_mri/reconstructor/model.pt')
    parser.add_argument("--in_chans", type=int, default=1, choices=[1, 2])
    parser.add_argument("--out_chans", type=int, default=1)
    parser.add_argument("--num_chans", type=int, default=16)
    parser.add_argument("--num_pool_layers", type=int, default=4)
    parser.add_argument("--drop_prob", type=float, default=0.)
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default='knee')
    parser.add_argument("--_data_location", type=str, default='../pg_mri/dataset/knee_singlecoil')
    parser.add_argument('--resolution', default=128, type=int, help='Resolution of images')
    parser.add_argument('--sample_rate', type=float, default=0.5,
                        help='Fraction of total volumes to include')
    parser.add_argument('--center_volume', type=int, default=1, choices=[0, 1], 
                        help='If set, only the center slices of a volume will be included in the dataset. This '
                             'removes the most noisy images from the data.')
    parser.add_argument('--acquisition', default=None,
                        help='Use only volumes acquired using the provided acquisition method. Options are: '
                             'CORPD_FBK, CORPDFS_FBK (fat-suppressed), and not provided (both used).')
    
    # Finish
    
    args = parser.parse_args()
    
    # transfer
    args.apply_attrs_padding = True if args.apply_attrs_padding else False
    args.budget = int(args.resolution/args.accelerate - int(args.resolution/args.low_frequency_mask_ratio))
    args.use_rms_prop = (args.use_rms_prop == 1)
    args.center_volume = (args.center_volume == 1)

    # save
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    with open(args.checkpoints_dir+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    return args

if __name__ == "__main__":
    args = build_args()
    set_random_seeds(args)
    torch.set_num_threads(args.num_workers)
    train(args)
