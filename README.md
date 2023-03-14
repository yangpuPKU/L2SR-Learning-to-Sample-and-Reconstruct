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

### 4. Train

### 5. Test


## Main Results

## Contact

Pu Yang [yang_pu@pku.edu.cn](mailto:yang_pu@pku.edu.cn)

Bin Dong [dongbin@math.pku.edu.cn](mailto:dongbin@math.pku.edu.cn)
