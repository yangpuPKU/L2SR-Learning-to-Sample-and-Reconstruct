import functools
import pathlib
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import gym
import time

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs

import activemri
import activemri.envs.envs as mri_env
from activemri.envs.envs import ActiveMRIEnv
from activemri.envs.util import create_data_range_dict
from activemri.data.pgmri_knee_data import SliceData, DataTransform
from activemri.models.reconstruction_model_utils import load_recon_model
from activemri.envs.torch_metrics import compute_ssim, compute_psnr
from activemri.data import transforms

from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
)
        

class SparseVecEnv(mri_env.ActiveMRIEnv, VecEnv):
    KSPACE_WIDTH = 368
    
    def __init__(self, options, mode='train'):
        mri_env.ActiveMRIEnv.__init__(self, 
            kspace_shape = [options.resolution, options.resolution],
            num_parallel_episodes = options.num_parallel_episodes,
            budget = options.budget,
            seed = options.random_seed,
        )
        self.opts = options
        self.mode = mode
        self.num_envs = options.num_parallel_episodes
        # Gym init
        self.observation_space = gym.spaces.Dict({
            'reconstruction_input': gym.spaces.Box(low=np.float(-np.infty), high=np.float(np.infty), shape=[1, self.opts.resolution, self.opts.resolution], dtype=np.float32), 
            'mask': gym.spaces.Box(low=np.float(0.), high=np.float(1.), shape=[self.opts.resolution], dtype=np.float)
        })
        self.action_space = gym.spaces.Discrete(self.opts.resolution)
        # setup
        self._setup()
        
    def _setup(self):
        self._has_setup = True
        # data handlers
        self._data_location = self.opts._data_location
        self._setup_data_handlers()
        # important parameters
        self._device = torch.device(self.opts.device)
        self.reward_metric = self.opts.reward_metric
        # mask
        mask_func = activemri.envs.util.import_object_from_str('activemri.envs.masks.sample_low_frequency_mask')
        mask_dict = {
            "width_dim": 1, 
            "max_width": self.opts.resolution, 
            "min_cols": self.opts.resolution / self.opts.low_frequency_mask_ratio / 2, 
            "max_cols": self.opts.resolution / self.opts.low_frequency_mask_ratio / 2, 
            "apply_attrs_padding": self.opts.apply_attrs_padding, 
            "centered": True, 
        }
        self._mask_func = functools.partial(mask_func, mask_dict)
        # reconstructor 
        # self._reconstructor = reconstructor_cls(**options)
        recon_args, recon_model = load_recon_model(self.opts, is_LOUPE=False)
        self._recon_args = recon_args
        self._reconstructor = recon_model
        self._reconstructor.eval()
        self._reconstructor.to(self._device)
        # transform
        self._transform = PPO_EndtoEnd_transform
        # data range
        if self.mode == 'train':
            self._current_data_range_dict = create_data_range_dict(self.opts, self._single_train_data_handler._data_loader)
        else:
            self._current_data_range_dict = create_data_range_dict(self.opts, self._current_data_handler._data_loader)
            
    def _setup_data_handlers(self):
        dataset = self._create_dataset()
        self._current_data_handler = mri_env.DataHandler(
            dataset,
            self._seed,
            batch_size=self.num_parallel_episodes,
            loops=self._num_loops_train_data if self.mode=='train' else 1,
            collate_fn=mri_env._env_collate_fn,
            mode=self.mode,
        )
        if self.mode == 'train':
            self._single_train_data_handler = mri_env.DataHandler(
                dataset,
                self._seed,
                batch_size=self.num_parallel_episodes,
                loops=1,
                collate_fn=mri_env._env_collate_fn,
            )
        
    def _create_dataset(self):
        root_path = pathlib.Path(self._data_location)
        train_path = root_path / "train"
        val_path = root_path / "val"
        test_path = root_path / "test"
        
        transform = DataTransform(resolution = self.opts.resolution)
        
        if self.mode == 'train':
            dataset = SliceData(
                root = train_path,
                transform = transform, 
                dataset = self.opts.dataset, 
                sample_rate=self.opts.sample_rate,
                acquisition=self.opts.acquisition,
                center_volume=self.opts.center_volume, 
            )
        elif self.mode == 'val':
            dataset = SliceData(
                root = val_path,
                transform = transform, 
                dataset = self.opts.dataset, 
                sample_rate=self.opts.sample_rate,
                acquisition=self.opts.acquisition,
                center_volume=self.opts.center_volume
            )
        elif self.mode == 'test':
            dataset = SliceData(
                root = test_path,
                transform = transform, 
                dataset = self.opts.dataset, 
                sample_rate=self.opts.sample_rate,
                acquisition=self.opts.acquisition,
                center_volume=self.opts.center_volume
            )
        print(f"length of {self.mode} dataset:", len(dataset))
        self.length_dataset = len(dataset)
        return dataset

    def seed(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        if self.mode == 'train':
            self._single_train_data_handler.seed(seed)
        else:
            self._current_data_handler.seed(seed)
    
    def _compute_obs(
        self, override_current_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        mask_to_use = (
            override_current_mask
            if override_current_mask is not None
            else self._current_mask
        )
        reconstruction_input = self._transform_wrapper(
            kspace=self._current_k_space,
            mask=mask_to_use,
            ground_truth=self._current_ground_truth,
        )

        obs = {
            "reconstruction_input": reconstruction_input,
            "mask": self._current_mask.clone().view(self._current_mask.shape[0], -1).cpu().squeeze().numpy(),
        }
        self.reconstruction_input = reconstruction_input.cpu().clone().detach()
        reconstruction_input = None  # de-referencing GPU tensors
        return obs

    def _compute_score(self, obs, done):
        if all(done):
            reconstruction_input = obs["reconstruction_input"]
            with torch.no_grad():
                reconstruction_input = reconstruction_input.to(self._device)
                reconstruction = self._reconstructor(reconstruction_input)
                self.reconstruction = reconstruction.cpu().numpy()
            score = self._compute_score_given_tensors(reconstruction, self._current_ground_truth.to(self._device), self._current_gt_mean.to(self._device), self._current_gt_std.to(self._device))
            del reconstruction_input
        else:
            score = {
                'mse': np.array([np.infty] * len(done)), 
                'nmse': np.array([np.infty] * len(done)), 
                'ssim': np.zeros(len(done)), 
                'psnr': np.zeros(len(done)), 
            }
        return score
    
    # @staticmethod
    def _compute_score_given_tensors(self, reconstruction: torch.Tensor, ground_truth: torch.Tensor, gt_mean=0, gt_std=1.0):
        data_range = self._current_data_range
        reconstruction = reconstruction.squeeze() * gt_std.unsqueeze(1).unsqueeze(1) + gt_mean.unsqueeze(1).unsqueeze(1)
        reconstruction = reconstruction.view(-1, 1, reconstruction.shape[-2], reconstruction.shape[-1])
        ground_truth = ground_truth.squeeze() * gt_std.unsqueeze(1).unsqueeze(1) + gt_mean.unsqueeze(1).unsqueeze(1)
        ground_truth = ground_truth.view(reconstruction.shape)
        mse = activemri.envs.util.compute_mse(reconstruction, ground_truth)
        nmse = activemri.envs.util.compute_nmse(reconstruction, ground_truth)
        ssim = compute_ssim(reconstruction, ground_truth, data_range=data_range, size_average=False).cpu().numpy()
        # psnr = activemri.envs.util.compute_psnr_torch(reconstruction, ground_truth).cpu()
        psnr = compute_psnr(self.opts, reconstruction, ground_truth, data_range=data_range).cpu().squeeze().numpy()

        return {"mse": mse, "nmse": nmse, "ssim": ssim*100, "psnr": psnr}
    
    
    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def reset(self, 
              return_info: bool = False, 
              data = None, 
    ):
        print('reset!!!!!!!!!!!')
        self._did_reset = True
        if data:
            kspace, _, ground_truth, gt_mean, gt_std, attrs, fname, slice_id = data
        else:
            try:
                kspace, _, ground_truth, gt_mean, gt_std, attrs, fname, slice_id = next(
                    self._current_data_handler
                )
            except:
                self._current_data_handler.reset()
                kspace, _, ground_truth, gt_mean, gt_std, attrs, fname, slice_id = next(
                    self._current_data_handler
                )

        self._current_ground_truth = torch.from_numpy(np.stack(ground_truth))
        self._current_gt_mean = torch.from_numpy(np.stack(gt_mean))
        self._current_gt_std = torch.from_numpy(np.stack(gt_std))
        self._current_data_range = torch.stack([self._current_data_range_dict[vol] for vol in fname])

        # Converting k-space to torch is better handled by transform,
        # since we have both complex and non-complex versions
        self._current_k_space = kspace

        self._transform_wrapper = functools.partial(
            self._transform, attrs=attrs, fname=fname, slice_id=slice_id  # padding left = 18, padding right = 350
        )
        kspace_shapes = [tuple(k.shape) for k in kspace]
        self._current_mask = self._mask_func(kspace_shapes, self._rng, attrs=attrs)
        obs = self._compute_obs()
        done = activemri.envs.masks.check_masks_complete(self._current_mask)
        if self._steps_since_reset >= self.budget:
            done = [True] * len(done)
        self._current_score = self._compute_score(obs=obs, done=done)
        self._steps_since_reset = 0

        metas = []
        for i in range(len(done)):
            metas.append({
                "fname": fname[i],
                "slice_id": slice_id[i],
                "current_score": self._current_score[self.reward_metric][i],
            })
        if return_info:
            return obs, metas
        else:
            return obs
        
    def step(
        self, action: Union[int, Sequence[int]], is_reset=True
    ) -> Tuple[Dict[str, Any], np.ndarray, List[bool], Dict]:
        if not self._did_reset:
            raise RuntimeError(
                "Attempting to call env.step() before calling env.reset()."
            )
        self._steps_since_reset += 1
        reward = None
        self._current_mask = activemri.envs.masks.update_masks_from_indices(
            self._current_mask, action
        )
        obs = self._compute_obs()
        done = activemri.envs.masks.check_masks_complete(self._current_mask)
        if self.budget and self._steps_since_reset >= self.budget:
            done = [True] * len(done)
        done = np.array(done)
        
        new_score = self._compute_score(obs=obs, done=done)
        reward = new_score[self.reward_metric] - self._current_score[self.reward_metric]
        if self.reward_metric in ["mse", "nmse"]:
            reward *= -1
        else:
            assert self.reward_metric in ["ssim", "psnr"]
        self._current_score = new_score
        
        infos = []
        for i in range(len(reward)):
            infos.append({"current_score": self._current_score[self.reward_metric][i], 
                          "current_score_psnr": self._current_score["psnr"][i]
            })
        if all(done) == True:
            for i in range(len(reward)):
                infos[i]["episode"] = {
                    'r': self._current_score[self.reward_metric][i], 
                    'l': self.budget,
                }
            if is_reset:
                obs = self.reset()
        return obs, reward, done, infos
    
    
    def step_without_reward(
        self, action: Union[int, Sequence[int]]
    ):
        self._steps_since_reset += 1
        self._current_mask = activemri.envs.masks.update_masks_from_indices(
            self._current_mask, action
        )
        obs = self._compute_obs()
        done = activemri.envs.masks.check_masks_complete(self._current_mask)
        if self.budget and self._steps_since_reset >= self.budget:
            done = [True] * len(done)
        done = np.array(done)
        return obs, done
    
    def step_async(self, actions):
        pass
    
    def step_wait(self):
        pass
    
    def get_attr(self, attr_name, indices=None):
        pass
    
    def set_attr(self, attr_name, value, indices=None):
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return True
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass
    
    
def PPO_EndtoEnd_transform(kspace=None, mask=None, ground_truth=None, attrs=None, fname=None, slice_id=None):
    batch_size = len(kspace)
    images = []
    for i in range(batch_size):
        image= _base_EndtoEnd_transform(
            kspace[i],
            mask[i],
            ground_truth[i],
            attrs[i],
        )
        images.append(image)
    return torch.stack(images)# .permute(0,3,1,2)
    
    

def _base_EndtoEnd_transform(
    kspace, mask, ground_truth, attrs, which_challenge="singlecoil"
):
    # kspace = fastmri_transforms.to_tensor(kspace)

    mask = mask[..., : kspace.shape[-2]]  # accounting for variable size masks
    masked_kspace = kspace * mask.unsqueeze(-1) + 0.0

    # inverse Fourier transform to get zero filled solution
    image = transforms.ifft2(masked_kspace)
    image = transforms.complex_abs(image)
    
    image, means, stds = transforms.normalize(image, dim=(-2, -1), eps=1e-11)
    image = image.clamp(-6, 6)
    
    return image.unsqueeze(0)