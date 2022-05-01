# SpanConv: A New Convolution via Spanning Kernel Space for Lightweight Pansharpening

**Homepage:**  

https://liangjiandeng.github.io/

https://chengjin.netlify.app/ 

https://tianjingzhang.github.io/

- Code for paper: " SpanConv: A New Convolution via Spanning Kernel Space for Lightweight Pansharpening."


# Citation

```bib
@article{SpanConv,
author = {Zhi-Xuan Chen, Cheng Jin, Tian-Jing Zhang, Xiao Wu, and Liang-Jian Deng},
title = {SpanConv: A New Convolution via Spanning Kernel Space for Lightweight},
conference = {International Joint Conferences on Artificial Intelligence},
volume = {},
pages = {},
year = {2022},
}
```

# Dependencies and Installation

- Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/))
- Pytorch 1.10.0
- NVIDIA GPU + CUDA
- Python packages: `pip install numpy scipy h5py`
- TensorBoard


# Dataset Preparation

The datasets used in this paper is WorldView-3 (can be downloaded [here](https://www.maxar.com/product-samples/)), QuickBird (can be downloaded [here](https://earth.esa.int/eogateway/catalog/quickbird-full-archive)). Due to the copyright of dataset, we can not upload the datasets, you may download the data and simulate them according to the paper.


# Get Started

-   Training and testing codes are in '[codes/](https://github.com/ChengJin-git/LPPN/blob/main/codes)'. Pretrained model on WorldView-3 can be found in '[codes/Weights/](https://github.com/ChengJin-git/LPPN/blob/main/codes/pretrained)'. All codes will be presented after the paper is completed published. Please refer to `codes/how-to-run.md` for detail description.


# Method

***Motivation:*** The standard convolution suffers from the large computational cost when the number of convolutional channels increase. However, in some situations or tasks, the information of the convolution kernels is redundant, as the following experimental result shows. In this paper, we propose a effective and efficient convolution module called SpanConv, which constructs a kernel space by several principal basis kernels.

<img src=".\figures\motivation.png" width = "60%" align="middle" />



***Proposed SpanConv:*** The following figure displays the generation detail of our proposed SpanKernel used in SpanConv. More details can be found in Sec. 3 of our paper.

<img src=".\figures\method.png" width = "70%" align="middle"/>



***Proposed LightNet:*** We design a network called LightNet to test the performance of SpanConv. Therefore, all convolutions used in the LightNet are all realized by SpanConv.

<img src=".\figures\net.png" width = "80%" align="middle"/>



***Visual Results:*** Visual comparisons of all the compared approaches on the reduced resolution Rio dataset (sensor: WorldView-3).

<img src=".\results\wv3_reduced.png" align="middle"/>



***Quantitative Results:*** The following quantitative results is generated from WorldView-3 reduced resolution datasets with 130 examples. 

<img src=".\results\wv3_reduced_table.png" width = "80%" align="middle"/>



# Contact

We are glad to hear from you. If you have any questions, please feel free to contact zhixuan_chen2022@163.com or open issues on this repository.

# License

This project is open sourced under GNU Affero General Public License v3.0.
