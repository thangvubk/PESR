# PESR
Official implementation for Perception-Enhanced Single Image Super-Resolution via Relativistic Generative Networks

## Dependencies
- Nvidia GPUs (4 GPUs for training or 1GPUs testing)
- At least 32G RAM 
- ``Python3``
- ``Pytorch 0.4``
- ``tensorboardX``
- ``tqdm``
- ``imageio``
- ``scipy``

## Dataset
- Train: DIV2K (800 2K-resolution images)
- Test: Set5, Set14, B100, Urban100, PIRM (100 self-val images), DIV2K (100 val images)

## Quick start
- Download test dataset and put into ``data/origin/`` directory
- Run ``python test.py --dataset <DATASET_NAME>``
- Results will be saved into ``results/`` directory

## Training
- Download train dataset and put into ``data/origin directory``
- Run ``python train.py``
- Models with be saved into ``check_point/`` direcory
- Observe tensorboard: Open another terminal window then ``tensorboard --logdir check_point``
- Enter: ``YOUR_IP:6006`` to your web browser.




