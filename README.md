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
  <a href='https://colab.research.google.com/'>
    <img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'>
  </a>
<br>
</h3>
</div>
 
  CPPF is a pure sim-to-real method that achieves 9D pose estimation in the wild. Our model is trained solely on ShapeNet synthetic models (without any real-world background pasting), and could be directly applied to real-world scenarios (i.e., NOCS REAL275, SUN RGB-D, etc.). CPPF achieves the goal by using only local $SE3$-invariant geometric features, and leverages a bottom-up voting scheme, which is quite different from previous end-to-end learning methods. Our model is robust to noise, and can obtain decent predictions even if only bounding box masks are provided.
  
# News
  
# Overview

This is the official code implementation of CPPF, including both training and testing. Inference on custom datasets is also supported.

## 
  
# Installation
You can run the following command to setup an environment, tested on Ubuntu 16.04:

```
conda create -n cppf python=3.8
conda install pytorch cudatoolkit=10.2 -c pytorch-lts
pip install tqdm opencv-python scipy matplotlib open3d==0.12.0 hydra-core pyrender cupy-cuda102 PyOpenGL-accelerate
CXX=g++-7 CC=gcc-7 pip install MinkowskiEngine==0.5.4 -v
```
Notice that we use pyrender with OSMesa support, you may need to install OSMesa after running ```pip install pyrender```, more details can be found [here](https://pyrender.readthedocs.io/en/latest/install/index.html).