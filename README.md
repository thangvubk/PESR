# PESR
Official implementation for Perception-Enhanced Single Image Super-Resolution via Relativistic Generative Networks
![PSNR vs PESR](https://github.com/thangvubk/PESR/blob/master/docs/PSNR_PESR.PNG)
## Dependencies
- Nvidia GPUs (training takes 1 day on 4 Titan Xp GPUs)
- At least 16G RAM 
- ``Python3``
- ``Pytorch 0.4``
- ``tensorboardX``
- ``tqdm``
- ``imageio``

## Datasets, models, and results
### Dataset
- Train: DIV2K (800 2K-resolution images)
- Valid (for visualization): DIV2K (100 val images), PIRM (100 self-val images)
- Test: Set5, Set14, B100, Urban100, PIRM (100 self-val images), DIV2K (100 val images)
- Download [train+val+test]() datasets
- Download [test only](https://drive.google.com/file/d/1zK9xo-rODnH5s6YlNKLvk4wP7-BZRuGc/view?usp=sharing) dataset
    
### Pretrained models
- Download [pretrained models]() including 1 PSNR-optimized model and 1 perception-optimized model
    
### Paper results
- Download [paper results]() in images of the test datasets

## Quick start
- Download test dataset and put into ``data/origin/`` directory
- Run ``python test.py --dataset <DATASET_NAME>``
- Results will be saved into ``results/`` directory

## Training
- Download train dataset and put into ``data/origin directory``
- Pretrain with L1 loss: ``python train.py --phase pretrain --learning_rate 1e-4``
- Finetune on pretrained model with GAN: ``python train.py``
- Models with be saved into ``check_point/`` direcory
- Observe tensorboard: Open another terminal window then ``tensorboard --logdir check_point``
- Enter: ``YOUR_IP:6006`` to your web browser.
- Tensorboard when finetuning on pretrained model should be similar to:

![Tensorboard](https://github.com/thangvubk/PESR/blob/master/docs/tensorboard.PNG)

![Tensorboard_imgs](https://github.com/thangvubk/PESR/blob/master/docs/tensorboard_img.PNG)

## Comprehensive testing
- Test perceptual model: follow [Quick start](#quick-start)
- Interpolate between perceptual model and PSNR model: ``python test.py --dataset <DATASET> --alpha <ALPHA>``  (with alpha being perceptual weight)
- Test perceptual quality: refer to [PIRM validation code](https://github.com/roimehrez/PIRM2018)

## Quantitative and Qualitative results
<p> RED and BLUE indicate best and second best respectively.</p>
<p align="center">
    <img src="https://github.com/thangvubk/PESR/blob/master/docs/quantitative.PNG">
    <img width="800" height="1200", src="https://github.com/thangvubk/PESR/blob/master/docs/qualitative.PNG">
</p>

## References
[EDSR-pytorch](https://github.com/thstkdgus35/EDSR-PyTorch)
[Relativistic-GAN](https://github.com/AlexiaJM/RelativisticGAN)
