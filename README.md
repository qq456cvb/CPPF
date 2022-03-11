<h1 align="center">
CPPF: Towards Robust Category-Level 9D Pose Estimation in the Wild
</h1>

<p align='center'>
<img align="center" src='images/intro.jpg' width='70%'> </img>
</p>

<div align="center">
<h3>
<a href="https://qq456cvb.github.io">Yang You</a>, Ruoxi Shi, Weiming Wang, Cewu Lu
<br>
<br>
CVPR 2022
<br>
<br>
<a href='https://arxiv.org/pdf/2203.03089.pdf'>
  <img src='https://img.shields.io/badge/Paper-PDF-orange?style=flat&logo=arxiv&logoColor=orange' alt='Paper PDF'>
</a>
<a href='https://qq456cvb.github.io/projects/cppf'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=googlechrome&logoColor=green' alt='Project Page'>
</a>
  <a href='#'>
    <img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'>
  </a>
<br>
</h3>
</div>
 
  CPPF is a pure sim-to-real method that achieves 9D pose estimation in the wild. Our model is trained solely on ShapeNet synthetic models (without any real-world background pasting), and could be directly applied to real-world scenarios (i.e., NOCS REAL275, SUN RGB-D, etc.). CPPF achieves the goal by using only local $SE3$-invariant geometric features, and leverages a bottom-up voting scheme, which is quite different from previous end-to-end learning methods. Our model is robust to noise, and can obtain decent predictions even if only bounding box masks are provided.
  
# Contents
- [Overview](#overview)
- [Installation](#installation)
- [Train on ShapeNet Objects](#train-on-shapenet-objects)
- [Pretrained Models](#pretrained-models)
- [Test on NOCS REAL275](#test-on-nocs-real275)
- [Test on SUN RGB-D](#test-on-sun-rgb-d)
- [Train on Your Own Object Collections](#train-on-your-own-object-collections)
# Overview

This is the official code implementation of CPPF, including both training and testing. Inference on custom datasets is also supported.
  
# Installation
You can run the following command to setup an environment, tested on Ubuntu 18.04:

```
conda create -n cppf python=3.8
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch-lts
pip install tqdm opencv-python scipy matplotlib open3d==0.12.0 hydra-core pyrender cupy-cuda102 PyOpenGL-accelerate OpenEXR
CXX=g++-7 CC=gcc-7 pip install MinkowskiEngine==0.5.4 -v
```
We use [Hydra](https://hydra.cc/) configuration system to run scripts.
Notice that we use pyrender with OSMesa support, you may need to install OSMesa after running ```pip install pyrender```, more details can be found [here](https://pyrender.readthedocs.io/en/latest/install/index.html).

# Train on ShapeNet Objects
<details>
<summary>Data Preparation</summary>

Download [ShapeNet v2](https://shapenet.org/) dataset and modify the ``shapenet_root`` key in ``config/config.yaml`` to point to the location of the dataset.

</details>

<details>
<summary>Train on NOCS REAL275 objects</summary>

To train on synthetic ShapeNet objects that appear in NOCS REAL275, run:
```
python train.py category=bottle,bowl,camera,can,laptop,mug -m
```

For laptops, an auxiliary segmentation is needed to ensure a unique pose. Please refer to <a href='#laptop-aux'>Auxiliary Segmentation for Laptops</a>/
</details>

<details>
<summary>Train on SUN RGB-D objects</summary>

To train on synthetic ShapeNet objects that appear in SUN RGB-D, run:
```
python train.py category=bathtub,bed,bookshelf,chair,sofa,table -m
```
</details>

<details>
<summary id='laptop-aux'>Auxiliary Segmentation for Laptops</summary>

For Laptops, geometry alone cannot determine the pose unambiguously, we rely on an auxiliary segmentation network that segments out the lid and the keyboard base.

To train the segmenter network, first download our Blender physically rendered laptop images from [Google Drive](https://drive.google.com/file/d/1gRHGt47nP9arDAu3hwnDNgfwJMxJYtCa/view?usp=sharing) and place it under ``data/laptop``. Then run the following command:
```
python train_laptop_aux.py
```
</details>


# Pretrained Models
Pretrained models for various ShapeNet categories can be downloaded from [Google Drive](https://drive.google.com/drive/folders/11wm5WHDjmSBfhng6emxCBBYZexmLoxLk?usp=sharing).
# Test on NOCS REAL275

<details>
<summary>Data Preparation</summary>

First download the detection priors from [Google Drive](https://drive.google.com/file/d/1cvGiXG_2ya8CMHss1IDobdL81qeODOrE/view?usp=sharing), which is used for evaluation with instance segmentation or bounding box masks. Put the directory under ``data/nocs_seg``.

Then download RGB-D images from [NOCS REAL275](http://download.cs.stanford.edu/orion/nocs/real_test.zip) dataset and put it under ``data/nocs``.

Place (pre-)trained models under ``checkpoints``.
</details>

<details>
<summary>Evaluate with Instance Segmentation Mask</summary>

First save inference outputs:
```
python nocs/inference.py
``` 

Then evaluate mAP: 
```
python nocs/eval.py | tee nocs/map.txt
```
</details>

<details>
<summary> Evaluate with Bounding Box Mask</summary>

First save inference outputs with bounding box mask enabled:
```
python nocs/inference.py --bbox_mask
``` 

Then evaluate mAP: 
```
python nocs/eval.py | tee nocs/map_bbox.txt
```
</details>

<details>
<summary> Zero-Shot Instance Segmentation and Pose Estimation</summary>
Coming soon.

</details>
# Test on SUN RGB-D
Coming soon.

# Train on Your Own Object Collections
Coming soon.