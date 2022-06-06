## DACL
This repo provides an official PyTorch implementation of "Learning a Domain-Agnostic Visual Representation for Autonomous Driving via Contrastive Loss" [[paper](https://arxiv.org/abs/2103.05902)]

### Overview
<p align="center">
	<img src="figure/figure1.png" alt="photo not available" width="80%" height="80%">
</p>

### Setup
It is recommended to create a new Anaconda virtual environment for reproduction or evaluation with pretrained models.


```bash
conda create -n dacl python==3.6.5
conda activate dacl
conda install pytorch=1.6.0 torchvision=0.7.0 -c pytorch
```

We ran our experiments with PyTorch 1.6.0, CUDA 11.0, Python 3.6.5 and Ubuntu 18.04. Usage of higher or lower version of PyTorch seems to be incompatible to our pre-trained model.

```bash
pip install pillow==7.2.0 opencv-contrib-python==4.4.0.42
```
If you do not use Anaconda environment, please use pip3 rather than pip for dependencies with Python3.


### Dataset

[vKITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/) is used as synthetic source domain data and [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) is used as realistic target domain data. A train & test split suggested in [Eigen et al.](https://arxiv.org/abs/1406.2283) is provided in datasplit folder and copy them in data folder.

```
/YOUR/OWN/ROOT
  |
  |----kitti 
         |----2011_09_26         
         |----2011_09_28        
         |----......... 
	 |-----train.txt
         |-----val.txt
	 |-----test.txt 	
	 
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
	       
	 |-----train.txt
         |-----val.txt
	 |-----test.txt 
```

### Train 
### Stage 1: Style Transfer

Train a bidirectional style transfer network via our providing codes, or download [pretrained models](https://drive.google.com/drive/folders/1S9FbJoiLJnHF6MrTr41Q2y2lPGS_nsi7?usp=sharing) from GASDA [github](https://github.com/sshan-zhao/GASDA).
```bash
python train.py --model style --root /YOUR/OWN/ROOT --gpu_ids 0 --batchSize 8 --loadSize 256 1024
```
### Stage 2: Unsupervised pretraining
We train the encoder of task-specific network with the results of style transfer network in stage 1 in a bidirectional way. We also provide [pretrained models](https://drive.google.com/drive/folders/1vDXdnQ8nRxOaixu3HkdanP0IfB39FEms?usp=sharing) of the encoders.

- Target to Source
```bash
python train.py --model ft_pretrain --root /YOUR/OWN/ROOT --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_tgt_premodel ./checkpoints/vkitti2kitti_style/best_net_G_Tgt.pth
```
- Source to Target
```bash
python train.py --model fs_pretrain --root /YOUR/OWN/ROOT --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_src_premodel ./checkpoints/vkitti2kitti_style/best_net_G_Src.pth
```

### Stage 3: Task-specific Finetuning
Finetune the task-specific network with pretrained encoders in stage 2. In depth estimation, we first train two unidirectional networks and finetune in a bidirectional way. Before training the depth estimation network in unidirectional flow, please place the pretrained encoders of networks provided by our repository or trained by yourself in the propoer folder. Depth estimation is trained with bidirectional flow wheras semantic segmentation is trained with unidirectional flow.
#### Depth Estimation
- Target to Source Depth  
```bash
python train.py --model ft --root /YOUR/OWN/ROOT --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_tgt_premodel ./checkpoints/vkitti2kitti_style/best_net_G_Tgt.pth --t_depth_premodel ./checkpoints/vkitti2kitti_ft_pretrain/best_net_G_Pretrain_T.pth
```
- Source to Target Depth
```bash
python train.py --model fs --root /YOUR/OWN/ROOT --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_src_premodel ./checkpoints/vkitti2kitti_style/best_net_G_Src.pth --s_depth_premodel ./checkpoints/vkitti2kitti_fs_pretrain/best_net_G_Pretrain_S.pth
```
- Bidirectional Depth
```bash
python train.py --freeze_bn --freeze_in --model depth --root /YOUR/OWN/ROOT --gpu_ids 0 --batchSize 4 --loadSize 192 640 --g_src_premodel ./checkpoints/vkitti2kitti_style/best_net_G_Src.pth --g_tgt_premodel ./checkpoints/vkitti2kitti_style/best_net_G_Tgt.pth --d_src_premodel ./checkpoints/vkitti2kitti_style/best_net_D_Src.pth --d_tgt_premodel ./checkpoints/vkitti2kitti_style/best_net_D_Tgt.pth --t_depth_premodel ./checkpoints/vkitti2kitti_ft/20_net_G_Depth_T.pth --s_depth_premodel ./checkpoints/vkitti2kitti_fs/20_net_G_Depth_S.pth 
```
#### Semantic Segmentation
- Source to Target Segmentation

```bash
python train.py --model seg --root /YOUR/OWN/ROOT --gpu_ids 0 --batchSize 8 --loadSize 192 640 --g_src_premodel ./cyclegan/G_Src.pth --s_seg_premodel ./checkpoints/vkitti2kitti_seg_pretrain/best_net_G_Pretrain_S.pth
```
### Evaluation
- Depth Estimation

Download [Pretrained Models](https://drive.google.com/drive/folders/1EKYUl1nh6s9tfEYJxlbjgGlk8RM7-ND4?usp=sharing) and place it in checkpoints/vkitti2kitti_depth folder for evaluating our best results.

```bash
python test_depth.py --root /YOUR/OWN/ROOT --test_datafile 'test.txt' --which_epoch best --gpu_ids 0 --batchSize 1 --loadSize 192 640
```
- Semantic Segmentation

Download [Pretrained Models](https://drive.google.com/file/d/1y07J2fENcCQ1hi93VzimUfwVhXA3MAiR/view?usp=sharing) and place it in checkpoints/vkitti2kitti_seg folder for evaluating our best results. We only support a single-GPU evaluation for semantic segmentation. For evaluating our method on KITTI semantic segmentation benchmark, you should download 200 images and its paired labels from KITTI semantic segmentation [here](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) and place it in kitti folder in root path.

```bash
python test_seg.py --root /YOUR/OWN/ROOT --which_epoch best --gpu_ids 0 --batchSize 1 --loadSize 192 640
```

### Reference

Some codes are sourced and modified from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [GASDA](https://github.com/sshan-zhao/GASDA) and [MoCo](https://github.com/facebookresearch/moco).

