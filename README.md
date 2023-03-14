# L2SR: Learning to Sample and Reconstruct for Accelerated MRI
This repository is the official implementation of: [L2SR: Learning to Sample and Reconstruct for Accelerated MRI](https://arxiv.org/abs/2212.02190) (arxiv, preprint).

## Abstract 
Accelerated MRI aims to find a pair of samplers and reconstructors to reduce acquisition time while maintaining the reconstruction quality. Most of the existing works focus on finding either sparse samplers with a fixed reconstructor or finding reconstructors with a fixed sampler. Recently, people have begun to consider learning samplers and reconstructors jointly. In this paper, we propose an alternating training framework for finding a good pair of samplers and reconstructors via deep reinforcement learning (RL). In particular, we propose a novel sparse-reward Partially Observed Markov Decision Process (POMDP) to formulate the MRI sampling trajectory. Compared to the existing works that utilize dense-reward POMDPs, the proposed sparse-reward POMDP is more computationally efficient and has a provable advantage over dense-reward POMDPs. We evaluate our method on fastMRI, a public benchmark MRI dataset, and it achieves state-of-the-art reconstruction performances.

## Getting Started

### 1. Environment
 - Create an environment
   ```
   conda create -n L2SR python=3.7
   ```
 - Use `requirement.txt` to install packages
   ```
   pip install -r requirement.txt
   ```
 - Roll back the versions of Torch and Torchvision (Important!)
   ```
   pip install torch==1.6.0 torchvision==0.7.0
   ```

### 2. Dataset 
We utilize the single-coil knee dataset and the multi-coil brain dataset from the [fastMRI Dataset](https://fastmri.org/). The data preprocessing is consistent with that in [PG-MRI](https://github.com/Timsey/pg_mri). We reserve $20\%$ of the training data as the test set. 

You should:
 - Download the the [fastMRI Dataset](https://fastmri.org/).
 - Use the File `./activemri/data/split` to split the dataset as
   ```
   <path_to_data>
     singlecoil_knee
       train
       val
       test
     multicoil_brain
       train
       val
       test
   ```

### 3. Pretrained Reconstruction Model
We use a U-net for reconstruction that is identical to the one used in the [this paper](https://arxiv.org/abs/2010.16262). The pre-trained model for this U-net can be trained using the [this code](https://github.com/Timsey/pg_mri). It should be noted that our settings are slightly different from those in the reference paper. Therefore, to obtain results identical to ours, the following parameters should be considered during pre-training. 
 - Use the negative of SSIM value as loss. 
 - Set sample_rate as 0.5 in knee dataset and 0.2 in brain dataset, as we do not use all slices for pre-training. 
 - Change the mixed heuristic sampling policy used in dense-reward POMDP to the terminal one used in sparse-reward POMDP. 

For specific details, please refer to the appendix section of the original paper and File `./pretrain`.

### 4. Train and Test
 - L2S: Learning to Sample. Use 
   ```
   sh train_sparse.sh
   ```
   for training, and
   ```
   sh test_sparse.sh
   ```
   for test. 

 - L2SR: Learning to Sample and Reconstruct. Notice that you should first do L2S to get a pre-trained policy and then do alternate training. Use
   ```
   sh train_alter.sh
   ```
   for traiing, and
   ```
   sh test_alter.sh
   ```
   for test. 

#### You can change parameters in these `.sh` files to train and test in other settings.  ####


### 5. Visulization 
See in the next version. 


## Main Results
See in the next version. 


## Contact

Pu Yang [yang_pu@pku.edu.cn](mailto:yang_pu@pku.edu.cn)

Bin Dong [dongbin@math.pku.edu.cn](mailto:dongbin@math.pku.edu.cn)
