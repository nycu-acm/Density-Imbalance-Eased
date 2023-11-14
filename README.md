# Density-Imbalance-Eased

### Install

This implementation uses Python 3.6, [Pytorch](http://pytorch.org/), [Pymesh](https://github.com/PyMesh/PyMesh), Cuda 10.1. 
```shell
# Copy/Paste the snippet in a terminal
git clone --recurse-submodules https://github.com/ThibaultGROUEIX/AtlasNet.git
cd AtlasNet 

#Dependencies
conda create -n atlasnet python=3.6 --yes
conda activate atlasnet
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch --yes
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

* **[Demo](./doc/demo.md)** :    ```python train.py --demo```
* **[Training](./doc/training.md)** :  ```python train.py --shapenet13```  *Monitor on  http://localhost:8890/*
