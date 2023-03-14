import argparse
from multiprocessing.util import is_exiting
import pathlib
import random
import numpy as np
import torch
import os
from activemri.envs.sparse_vecenvs import SparseVecEnv
from activemri.feature_extractor import extractor
from stable_baselines3 import A2C
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def _save_image(image, image_pth, is_report=True):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(image_pth+'.png', bbox_inches='tight', pad_inches=0)
    if is_report:
        print('Successfully save image to: ' + image_pth)
    plt.close()
    np.save(image_pth+'.npy', image)

def test(args):
    #####################
    #---- build env ----#
    #####################
    env = SparseVecEnv(args, mode='test')
    print('--- Successfully load environment ---\n')
    print("Number of available actions:", env.action_space.n)

    
    ##################
    #---- policy ----#
    ##################
    policy_kwargs = {
        'net_arch': dict(), 
        'features_extractor_class': extractor.PPO_Extractor, 
        'features_extractor_kwargs': {'opts': args}, 
    }
    
    
    ################
    #---- test ----#
    ################
    model = A2C(
        policy = "MultiInputPolicy", 
        env = env, 
        policy_kwargs = policy_kwargs, 
        device = args.device, 
        verbose = 1, 
    )
    model.policy.action_net = extractor.Action_net().to(model.device)
    print(model.policy)
    model.set_parameters(args.training_dir, device=model.device)

    location = args.training_dir.rfind('/')

    action_array = np.zeros((env.length_dataset, env.budget))
    total_score = 0.
    ssim_score_list = []
    total_score_psnr = 0.
    psnr_score_list = []
    cnt = 0
    # obs = env.reset()
    for _ in range(len(env._current_data_handler._data_loader)):
        obs = env.reset()
        done = False
        timestep = 0
        while not done:
            action = model.predict(obs, deterministic=True)
            action_array[cnt:cnt+len(action[0]), timestep] = action[0]
            timestep += 1
            obs, rewards, dones, metas = env.step(action[0], is_reset=False)
            done = all(dones)
        # visualization
        try:
            os.mkdir(args.training_dir[:location]+'/visualization')
        except:
            pass
        for meta in metas:
            # if cnt % 1000 == 0:
            #     mask = torch.stack([env._current_mask[cnt%64,0]]*128)
            #     zero_filled_image = env.reconstruction_input[cnt%64, 0]
            #     kspace = torch.log(env._current_k_space[cnt%64].norm(dim=-1))
            #     recon_image = env.reconstruction[cnt%64, 0]
            #     gt = env._current_ground_truth[cnt%64]
            #     print(mask.shape, zero_filled_image.shape, kspace.shape, recon_image.shape, gt.shape)
            #     _save_image(mask, args.training_dir[:location]+'/visualization/mask_'+str(cnt+1))
            #     _save_image(zero_filled_image, args.training_dir[:location]+'/visualization/zf_'+str(cnt+1))
            #     _save_image(kspace, args.training_dir[:location]+'/visualization/kspace_'+str(cnt+1))
            #     _save_image(recon_image, args.training_dir[:location]+'/visualization/recon_'+str(cnt+1))
            #     _save_image(gt, args.training_dir[:location]+'/visualization/gt_'+str(cnt+1))

            cnt += 1
            print(f'{cnt}:', meta['current_score'], meta["current_score_psnr"])
            ssim_score_list.append(meta['current_score'])
            psnr_score_list.append(meta["current_score_psnr"])
            total_score += meta['current_score']
            total_score_psnr += meta["current_score_psnr"]
        
    print('num of test set:', cnt)
    avg_score = total_score / cnt
    avg_score_psnr = total_score_psnr / cnt
    print(f'score {args.reward_metric} = {avg_score}')

    # save score
    score_array = np.array([ssim_score_list, psnr_score_list])
    np.save(args.training_dir[:location]+'/score_array.npy', score_array)

    # write 
    # with open(args.training_dir[:location]+'/test.txt', 'a') as f:
    #     f.write(args.recon_model_checkpoint + ' SSIM: ' + str(round(np.array(ssim_score_list).mean(), 2))
    #                                                     + ' +/- ' + str(round(np.array(ssim_score_list).std(), 2))
    #                                         + '; PSNR: ' + str(round(np.array(psnr_score_list).mean(), 2))
    #                                                     + ' +/- ' + str(round(np.array(psnr_score_list).std(), 2)) + '\n')
            

def set_random_seeds(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    

def build_args():
    parser = argparse.ArgumentParser(description='MRI Reconstruction Example')
    
    parser.add_argument("--env_type", type=str, default='sparse')
    
    parser.add_argument("--accelerate", type=int, default=4)
    parser.add_argument("--num_parallel_episodes", type=int, default=4)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--training_dir", type=str, default=None)
    parser.add_argument(
        "--ppo_model_type",
        type=str,
        default="simple_mlp",
    )
    parser.add_argument(
        "--reward_metric",
        type=str,
        choices=["mse", "ssim", "nmse", "psnr"],
        default="ssim",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ppo_batch_size", type=int, default=16)
    
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
    
    # transfer
    args = parser.parse_args()
    args.apply_attrs_padding = True if args.apply_attrs_padding else False
    args.budget = int(args.resolution/args.accelerate - int(args.resolution/args.low_frequency_mask_ratio))
    args.center_volume = (args.center_volume == 1)
    
    return args

if __name__ == "__main__":
    args = build_args()
    set_random_seeds(args)
    torch.set_num_threads(8)

    test(args)