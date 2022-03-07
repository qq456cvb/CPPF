<h1 align="center">
CPPF: Towards Robust Category-Level 9D Pose Estimation in the Wild
</h1>

<p align='center'>
<img align="center" src='images/intro.jpg' width='80%'> </img>
</p>

<div align="center">
<h3>
<a href="https://qq456cvb.github.io">Yang You</a>, Ruoxi Shi, Weiming Wang, Cewu Lu
<br>
<br>
CVPR 2022
<br>
<br>
<a href='https://qq456cvb.github.io/files/cppf.pdf'>
  <img src='https://img.shields.io/badge/Paper-PDF-orange?style=flat&logo=arxiv&logoColor=orange' alt='Paper PDF'>
</a>
<a href='https://qq456cvb.github.io/cppf'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=googlechrome&logoColor=green' alt='Project Page'>
</a>
  <a href='https://colab.research.google.com/'>
      <img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'>
    </a>
<br>
  </div>
 
 <img>
  CPPF is a pure sim-to-real method that achieves 9D pose estimation in the wild. Our model is trained solely on ShapeNet synthetic models (without any real-world background pasting), and could be directly applied to real-world scenarios (i.e., NOCS REAL275, SUN RGB-D, etc.). CPPF achieves the goal by using only local *SE3*-invariant geometric features, and leverages a bottom-up voting scheme, which is quite different from previous end-to-end learning methods. Our model is robust to noise, and can obtain decent predictions even if only bounding box masks are provided.
  
# News
  
# Overview

This is the official code implementation of CPPF, including both training and testing. Inference on custom datasets is also supported.

## 
  
# Installation

Code coming soon