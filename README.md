# Density-Imbalance-Eased

This is the official code implementing the paper "Density-Imbalance-Eased LiDAR Point Cloud Upsampling via Feature Consistency Learning" IEEE Transactions on Intelligent Vehicles (T-IV 2022)

![alt text](https://github.com/nycu-acm/Density-Imbalance-Eased/blob/main-PUGAN/images/Overview.png?raw=true)

### Install

This implementation uses Python 3.7, [Pytorch](http://pytorch.org/), [Pymesh](https://github.com/PyMesh/PyMesh), Cuda 10.1. 
```shell
# Copy/Paste the snippet in a terminal
git clone https://github.com/nycu-acm/Density-Imbalance-Eased.git
cd Density-Imbalance-Eased

#Dependencies
conda create -n Density python=3.7
conda activate Density
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user --requirement  requirements.txt # pip dependencies
```

##### Optional : Compile Chamfer (MIT) + Metro Distance (GPL3 Licence)
```shell
# Copy/Paste the snippet in a terminal
python auxiliary/ChamferDistancePytorch/chamfer3D/setup.py install #MIT
cd auxiliary
git clone https://github.com/ThibaultGROUEIX/metro_sources.git
cd metro_sources; python setup.py --build # build metro distance #GPL3
cd ../..
```

### Usage

* **[Training](./doc/training.md)** :  ```python train.py --shapenet13```  *Monitor on  http://localhost:8890/*

### Quantitative Results on the PUGAN Dataset

| Method                 | CD (10<sup>-3</sup>) | HD (10<sup>-3</sup>) | P2F (10<sup>-3</sup>) | Time (ms) |
| ---------------------- | ---- | ----   | ----- |-------     |
| PU-Net | 0.520 | 3.952   | 4.633  | 10.04       |
| MPU   | 0.279 | 4.959   | 2.860  | 10.86    |
| PU-GAN | 0.242 | 3.020 | 1.973 | 14.28      |
| PU-GCN  | 0.260 | 2.919 |  2.501  | 8.83      |
| LiPUpNet (Ours)  | 0.237 | 3.655 |  1.907  | 11.10      |

### Citations
```
@article{CitekeyArticle,
  author   = "Tso-Yuan Chen, Ching-Chun Hsiao, Ching-Chun Huang",
  title    = "Density-Imbalance-Eased LiDAR Point Cloud Upsampling via Feature Consistency Learning",
  journal  = "IEEE Transactions on Intelligent Vehicles (T-IV)",
  year     = 2022,
  volume   = "8",
  number   = "4",
  pages    = "2875--2997",
}
```

