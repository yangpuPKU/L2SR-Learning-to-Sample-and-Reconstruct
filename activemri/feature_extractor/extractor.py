import numpy as np
import torch
from torch import nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from .Unet_encoder import Unet_encoder

class Extractor(BaseFeaturesExtractor): 
    def __init__(self, 
                 observation_space: gym.spaces.Dict, 
                 opts, 
    ):
        super(Extractor, self).__init__(observation_space, features_dim=1)
        
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'mask':
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)
            if key != 'mask' and len(subspace.shape) >= 2:
                extractors[key] = Unet_encoder(resolution=opts.resolution, chans=opts.num_chans)
                total_concat_size += opts.resolution
        
        self.extractors = nn.ModuleDict(extractors)
        
        self._features_dim = total_concat_size
        
    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []
        
        for key, extractor in self.extractors.items():
            if key != 'mask' and observations[key].shape[-1] == 2:
                observations[key] = observations[key].permute(0,3,1,2).norm(dim=1).unsqueeze(1)
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
    
    
class Action_net(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, features):
        bs, double_action_dim = features.shape
        action_dim = int(double_action_dim / 2)
        output = features[:, action_dim:] - features[:, :action_dim]*1e10
        return output