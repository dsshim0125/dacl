## Learning a Domain-Agnostic Representation for Autonomous Driving via Contrastive Loss
### IEEE IROS 2021 Submission
This repo provides an official PyTorch implementation of DACL submitted in IEEE IROS 2021 and offers pretrained encoders, and depth estimation and semantic segmentation networks.

### Overview
<p align="center">
	<img src="figure/figure1.png" alt="photo not available" width="80%" height="80%">
</p>

### Setup
It is recommended to create a new Anaconda virtual environment for reproduction or evaluation with pretrained models.


```bash
conda create -n dacl python==3.6.5
conda activate dacl
conda install pytorch=1.6.0 torchvision=0.5.0 -c pytorch
```

We ran our experiments with PyTorch 1.6.0, CUDA 10.2, Python 3.5.6 and Ubuntu 18.04. Usage of higher or lower version of PyTorch seems to be incompatible to our pre-trained model.

```bash
pip install pillow==7.2.0 opencv-contrib-python==4.4.0.42
```
If you do not use Anaconda environment, please use pip3 rather than pip for dependencies with Python3.


### Dataset

[vKITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/) is used as synthetic source domain data and [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) is used as realistic target domain data. A train & test split suggested in [Eigen et al.](https://arxiv.org/abs/1406.2283) is provided in datasplit folder.

```
<DATA_ROOT>
  |
  |----kitti 
         |----2011_09_26         
         |----2011_09_28        
         |----......... 
	 
  |----vkitti 
         |----rgb        
               |----0006              
               |-----.......  
	       
         |----depth       
               |----0006        
               |----.......
	       
         |----semantic_label       
               |----0006        
               |----.......
```

### Train 
### Stage 1: Style Transfer

Train a bidirectional style transfer network via our providing codes, or download [pretrained models](https://onedrive.live.com/?authkey=%21AJorokRKW4ATgZ8&id=CC387DDA3ECA5EAF%21113&cid=CC387DDA3ECA5EAF) from GASDA github.
```bash
python train.py --model style --gpu_ids 0 --batchSize 8 --loadSize 256 1024
```
### Stage 2: Unsupervised pretraining
We train the encoder of task-specific network with the results of style transfer network in stage 1 in a bidirectional way. We also provide pretrained models of the encoders for [fs](https://drive.google.com/file/d/1wGXk9mSpLfrkSo1wzvjhGb_6BKQLbT4H/view?usp=sharing) and [ft](https://drive.google.com/file/d/1HUYx-oBw645oGrNlMF33Sp2O_46hl2lV/view?usp=sharing).

- Source to Target
```bash
python train.py --model ft_pretrain --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_tgt_premodel ./cyclegan/G_Tgt.pth
```
- Target to Source
```bash
python train.py --model fs_pretrain --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_src_premodel ./cyclegan/G_Src.pth
```
### Stage 3: Task-specific Finetuning
Finetune the task-specific network with pretrained encoders in stage 2. In depth estimation, we first train two unidirectional networks and finetune in a bidirectional way. Depth estimation is trained with bidirectional flow wheras semantic segmentation is trained with unidirectional flow.
#### Depth Estimation
- Target to Source Depth  
```bash
python train.py --model ft --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_tgt_premodel ./cyclegan/G_Tgt.pth --t_depth_premodel ./checkpoints/vkitti2kitti_ft_pretrain/best_net_G_Pretrain_T.pth
```
- Source to Target Depth
```bash
python train.py --model fs --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_src_premodel ./cyclegan/G_Src.pth --s_depth_premodel ./checkpoints/vkitti2kitti_fs_pretrain/best_net_G_Pretrain_S.pth
```
- Bidirectional Depth
```bash
python train.py --freeze_bn --freeze_in --model depth --gpu_ids 0 --batchSize 4 --loadSize 192 640 --g_src_premodel ./cyclegan/G_Src.pth --g_tgt_premodel ./cyclegan/G_Tgt.pth --d_src_premodel ./cyclegan/D_Src.pth --d_tgt_premodel ./cyclegan/D_Tgt.pth --t_depth_premodel ./checkpoints/vkitti2kitti_ft/20_net_G_Depth_T.pth --s_depth_premodel ./checkpoints/vkitti2kitti_fs/20_net_G_Depth_S.pth 
```
#### Semantic Segmentation
- Source to Target Segmentation
```bash
python train.py --model seg --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_src_premodel ./cyclegan/G_Src.pth --s_depth_premodel ./checkpoints/vkitti2kitti_fs_pretrain/best_net_G_Depth_S.pth
```
### Evaluation
- Depth Estimation
```bash
python test.py --test_datafile 'test.txt' --which_epoch best --model depth --gpu_ids 0 --batchSize 1 --loadSize 192 640
```

### Reference

Some codes are sourced and modified from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [GASDA](https://github.com/sshan-zhao/GASDA) and [MoCo](https://github.com/facebookresearch/moco).

