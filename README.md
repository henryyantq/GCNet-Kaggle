# [Leveraging Inlier Correspondences Proportion for Point Cloud Registration](https://arxiv.org/pdf/2201.12094.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighborhood-aware-geometric-encoding-network/point-cloud-registration-on-3dmatch-at-least-2)](https://paperswithcode.com/sota/point-cloud-registration-on-3dmatch-at-least-2?p=neighborhood-aware-geometric-encoding-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighborhood-aware-geometric-encoding-network/point-cloud-registration-on-3dlomatch-10-30)](https://paperswithcode.com/sota/point-cloud-registration-on-3dlomatch-10-30?p=neighborhood-aware-geometric-encoding-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighborhood-aware-geometric-encoding-network/point-cloud-registration-on-3dmatch-at-least-1)](https://paperswithcode.com/sota/point-cloud-registration-on-3dmatch-at-least-1?p=neighborhood-aware-geometric-encoding-network)


Paper "Neighborhood-aware Geometric Encoding Network for Point Cloud Registration" was renamed to "Leveraging Inlier Correspondences Proportion for Point Cloud Registration" (NgeNet -> GCNet). 

## Use in Kaggle/Colab notebook environment (with GPU accelerator on)

### Prerequisition Installation

```sh
# Configure system environment
!apt-get update -y
!apt-get upgrade -y
!apt-get dist-upgrade -y
!apt-get install build-essential python-dev python-setuptools python-pip python-smbus -y
!apt-get install libncursesw5-dev libgdbm-dev libc6-dev -y 
!apt-get install zlib1g-dev libsqlite3-dev tk-dev -y
!apt-get install libssl-dev openssl -y
!apt-get install libffi-dev -y
!apt-get clean

# Deploy Python 3.8
%cd /kaggle/working
!wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
!tar -xzf Python-3.8.12.tgz

%cd /kaggle/working/Python-3.8.12
!./configure --enable-optimizations
!make
```

```sh
# Deploy GCNet
%cd /kaggle/working
!rm -rf ./*
!git clone https://github.com/henryyantq/GCNet-Kaggle

%cd /kaggle/working/GCNet-Kaggle
!python3.8 -m venv venv
!source venv/bin/activate && pip install -U pip
!source venv/bin/activate && pip install -U setuptools wheel
!source venv/bin/activate && pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
!source venv/bin/activate && pip install -r requirements.txt
!pip cache purge
```

### Inference with GCNet

```sh
# Upload the pretrained model checkpoint as well as your point cloud data (plyfiles) to the "Data" section of your Kaggle workspace
%cd /kaggle/working/GCNet-Kaggle
!source venv/bin/activate && cd cpp_wrappers && sh compile_wrappers.sh
!source venv/bin/activate && python demo.py --src_path "/kaggle/input/path_to_your_src_plyfile.ply" --tgt_path "/kaggle/input/path_to_your_target_plyfile.ply" --checkpoint "/kaggle/input/path_to_your_GCNet_3dmatch.pth" --voxel_size 0.025 --npts 20000 --no_vis
```

## [Pretrained weights (Optional)]

Download pretrained weights for 3DMatch, 3DLoMatch, Odometry KITTI and MVP-RG from [GoogleDrive](https://drive.google.com/drive/folders/1JDn6zQfLdZfAVVboXRrrrCVRo48pRjyW?usp=sharing) or [BaiduDisk](https://pan.baidu.com/s/18G_Deim1UlSkY8wWoOiwnw) (pwd: `vr9g`). **For personal use, the "3dmatch.pth" should be enough, normally.**

## Personal data

### Personal data (with the same voxel size as 3DMatch)

```
python demo.py --src_path demo_data/src1.ply --tgt_path demo_data/tgt1.ply --checkpoint your_path/3dmatch.pth  --voxel_size 0.025 --npts 20000
```
![](demo_data/my_data1.png)

### Personal data (with different voxel size from 3DMatch)

```
python demo.py --src_path demo_data/src2.ply  --tgt_path demo_data/tgt2.ply --checkpoint your_path/3dmatch.pth --voxel_size 3 --npts 20000
```
![](demo_data/my_data2.png)

Set an appropriate `voxel_size` for your test data. If you want to test on point cloud pair with **large amount of points**, please **set a large `voxel_size` according to your data**.

## Citation

```
@article{zhu2022leveraging,
  title={Leveraging Inlier Correspondences Proportion for Point Cloud Registration},
  author={Zhu, Lifa and Guan, Haining and Lin, Changwei and Han, Renmin},
  journal={arXiv preprint arXiv:2201.12094},
  year={2022}
}
```

## Acknowledgements

Thanks for the open source code [OverlapPredator](https://github.com/overlappredator/OverlapPredator), [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch), [KPConv.pytorch](https://github.com/XuyangBai/KPConv.pytorch), [FCGF](https://github.com/chrischoy/FCGF), [D3Feat.pytorch](https://github.com/XuyangBai/D3Feat.pytorch), [MVP_Benchmark](https://github.com/paul007pl/MVP_Benchmark) and [ROPNet](https://github.com/zhulf0804/ROPNet).
