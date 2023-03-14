import argparse
import json
import shutil
import pathlib
import random
from matplotlib.style import use
import numpy as np
import torch
import os
from activemri.envs.sparse_vecenvs import SparseVecEnv
from activemri.feature_extractor import extractor
from activemri.envs.torch_metrics import compute_ssim, compute_psnr
from stable_baselines3.common import env_checker
from stable_baselines3 import A2C
import matplotlib
import time
matplotlib.use('Agg')

def save_model(args, exp_dir, alter, epoch, model, optimizer, best_dev_loss, is_new_best):
    exp_dir = pathlib.Path(exp_dir)
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / f'model_{alter}.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / f'model_{alter}.pt', exp_dir / f'best_model_{alter}.pt')


def build_optim(args, env):
    optimizer = torch.optim.Adam(env._reconstructor.parameters(), args.lr_recon, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    return optimizer, scheduler


def train_recon_epoch(args, model, env, optimizer, epoch):
    start_iter = time.perf_counter()
    true_avg_loss = 0.

    for iter in range(len(env._single_train_data_handler._data_loader)):
        # sample with our policy
        obs = env.reset()
        done = False
        while not done:
            # action = model.predict(obs, deterministic=True)
            action = model.predict(obs, deterministic=args.is_determin)
            obs, dones = env.step_without_reward(action[0])
            done = all(dones)
        # train reconstructor
        zf = obs["reconstruction_input"]
        zf = zf.to(args.device)  # [bs, 1, resolution, resolution]
        unnorm_recon = env._reconstructor(zf)  # [bs, 1, resolution, resolution]
        gt_mean = env._current_gt_mean.to(args.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        gt_std = env._current_gt_std.to(args.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        recon = unnorm_recon * gt_std + gt_mean
        target = env._current_ground_truth.to(args.device).unsqueeze(1) * gt_std + gt_mean
        optimizer.zero_grad()
        loss = -compute_ssim(recon, target, size_average=True, data_range=env._current_data_range)
        loss.backward()
        optimizer.step()
        true_avg_loss = (true_avg_loss * iter + loss.mean()) / (iter + 1)

        if iter % 100 == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(env._single_train_data_handler._data_loader):4d}] '
                f'Loss = {loss.item():.4g} TrueAvgLoss = {true_avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return true_avg_loss


def evaluate_loss(args, model, reconstructor, alter=0, partition='test'):
    dev_env = SparseVecEnv(args, mode=partition)
    dev_env._reconstructor.load_state_dict(reconstructor.state_dict())
    dev_env._reconstructor.eval()
    cnt = 0
    total_score = 0.
    obs, meta = dev_env.reset(return_info=True)
    for _ in range(len(dev_env._current_data_handler._data_loader)):
        # obs, meta = env.reset(return_info=True)
        done = False
        timestep = 0
        while not done:
            action = model.predict(obs, deterministic=True)
            timestep += 1
            obs, rewards, dones, metas = dev_env.step(action[0])
            done = all(dones)
        for meta in metas:
            cnt += 1
            total_score += meta['current_score']
        
    avg_score = total_score / cnt
    print(partition + f'score {args.reward_metric} = {avg_score}')
    if partition == 'test':
        with open(args.checkpoints_dir+'/test.txt', 'a') as f:
            f.write(f'Alter: {alter}; SSIM: {str(round(avg_score, 2))} \n')
    return avg_score


def train_recon(args, model, env, alter):
    # reload reconstructor
    for parameter in env._reconstructor.parameters():
        parameter.requires_grad = True
    env._reconstructor.train()
    # optimizer
    optimizer, scheduler = build_optim(args, env)
    # train
    best_dev_score = 0.
    for epoch in range(args.num_epochs):
        avg_loss = train_recon_epoch(args, model, env, optimizer, epoch)
        dev_score = evaluate_loss(args, model, env._reconstructor, partition='val')
        scheduler.step()
        is_new_best = dev_score > best_dev_score
        best_dev_score = min(best_dev_score, dev_score)
        save_model(env._recon_args, args.checkpoints_dir, alter, epoch, env._reconstructor, optimizer, best_dev_score, is_new_best)
        # with open(args.checkpoints_dir+'/test.txt', 'a') as f:
        #     f.write(f'Alter: {alter}; SSIM: {str(round(dev_score, 2))} \n')
    # back to the best
    checkpoint = torch.load(args.checkpoints_dir+f'/best_model_{alter}.pt', map_location='cpu')
    env._reconstructor.load_state_dict(checkpoint['model'])
    del checkpoint
    # rehabilitation
    for parameter in env._reconstructor.parameters():
        parameter.requires_grad = False
    env._reconstructor.eval()


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
        'features_extractor_class': extractor.PPO_Extractor, 
        'features_extractor_kwargs': {'opts': args}, 
    }
    
    #################
    #---- train ----#
    #################
    model = A2C(
        policy = 'MultiInputPolicy', 
        env = train_env, 
        learning_rate = args.lr_RL, 
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
    model.policy.action_net = extractor.Action_net().to(model.device)
    print(model.policy)
    print('n_envs:', model.n_envs)

    # learn
    for alter in range(1, args.num_alters+1):
        # train sampler
        if alter == 1:
            # resume
            if args.training_dir:
                model.set_parameters(args.training_dir, device=model.device)
            else:
                model.learn(total_timesteps=args.num_train_steps, log_interval=100, reset_num_timesteps=False)
        else: 
            # model.learning_rate = args.lr_RL / args.lr_decay
            # model.learning_rate = model.learning_rate / args.lr_decay
            # model.ent_coef = model.ent_coef / args.lr_decay
            args.lr_recon = args.lr_recon / args.lr_decay
            model._setup_lr_schedule()
            model.learn(total_timesteps=args.num_train_steps/5, log_interval=100, reset_num_timesteps=False)
        evaluate_loss(args, model, train_env._reconstructor, float(alter)-0.5, 'test')
        model.save(args.checkpoints_dir+f'/final_model_{alter}')
        # train reconstructor
        train_recon(args, model, train_env, alter)
        evaluate_loss(args, model, train_env._reconstructor, alter, 'test')
    
    
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

    # alter training
    parser.add_argument("--num_alters", type=int, default=5)

    # recon training parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr_recon", type=float, default=0.001)
    parser.add_argument("--is_determin", type=int, default=1, choices=[0, 1])
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument('--lr_step_size', type=int, default=5)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    
    # A2C training parameters
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--use_rms_prop", type=int, choices=[0, 1], default=1)
    parser.add_argument("--lr_RL", type=float, default=0.0003)
    parser.add_argument("--lr_decay", type=int, default=10)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--update_timestep", type=int, default=1000)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Podel parameters
    parser.add_argument('--fc_size', default=256, type=int, help='Size (width) of fully connected layer(s).')
    
    # Mask parameters
    parser.add_argument('--accelerations', nargs='+', default=[8], type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for '
                             'each volume.')
    parser.add_argument("--low_frequency_mask_ratio", type=int, default=8)
    parser.add_argument("--apply_attrs_padding", type=int, default=0, choices=[0, 1])
    
    # Reconstructor parameters
    parser.add_argument("--recon_model_checkpoint", type=str, default='/home/yangpu/MRI/pg_mri/reconstructor/model.pt')
    parser.add_argument("--in_chans", type=int, default=1, choices=[1, 2])
    parser.add_argument("--out_chans", type=int, default=1)
    parser.add_argument("--num_chans", type=int, default=16)
    parser.add_argument("--num_pool_layers", type=int, default=4)
    parser.add_argument("--drop_prob", type=float, default=0.)
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default='knee')
    parser.add_argument("--_data_location", type=str, default='/home/yangpu/MRI/pg_mri/dataset/knee_singlecoil')
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
    args.is_determin = (args.is_determin == 1)
    args.center_volume = (args.center_volume == 1)
    args.checkpoints_dir = args.checkpoints_dir + f'/seed={args.random_seed}_lrRL={args.lr_RL}_ent={args.entropy_coef}_lrRecon={args.lr_recon}_lrstepsize={args.lr_step_size}_step={args.num_train_steps}'

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