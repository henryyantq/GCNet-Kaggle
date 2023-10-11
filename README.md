# [Leveraging Inlier Correspondences Proportion for Point Cloud Registration](https://arxiv.org/pdf/2201.12094.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighborhood-aware-geometric-encoding-network/point-cloud-registration-on-3dmatch-at-least-2)](https://paperswithcode.com/sota/point-cloud-registration-on-3dmatch-at-least-2?p=neighborhood-aware-geometric-encoding-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighborhood-aware-geometric-encoding-network/point-cloud-registration-on-3dlomatch-10-30)](https://paperswithcode.com/sota/point-cloud-registration-on-3dlomatch-10-30?p=neighborhood-aware-geometric-encoding-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neighborhood-aware-geometric-encoding-network/point-cloud-registration-on-3dmatch-at-least-1)](https://paperswithcode.com/sota/point-cloud-registration-on-3dmatch-at-least-1?p=neighborhood-aware-geometric-encoding-network)


Paper "Neighborhood-aware Geometric Encoding Network for Point Cloud Registration" was renamed to "Leveraging Inlier Correspondences Proportion for Point Cloud Registration" (NgeNet -> GCNet).

## Use in Kaggle/CoLab notebook environment 

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
%cd /kaggle/working/GCNet-Kaggle
!source venv/bin/activate && cd cpp_wrappers && sh compile_wrappers.sh
!source venv/bin/activate && python demo.py --src_path "/kaggle/input/path_to_your_src_plyfile.ply" --tgt_path "/kaggle/input/path_to_your_target_plyfile.ply" --checkpoint "/kaggle/input/path_to_your_GCNet_3dmatch.pth" --voxel_size 0.025 --npts 20000
```

## Results (saved in reg_results/3DMatch*-pred)

- Recall on 3DMatch and 3DLoMatch (correspondences RMSE below 0.2)

    | Dataset | npairs | Scene Recall (%) | Pair Recall (%) |
    | :---: | :---: | :---: | :---: |
    | 3DMatch | 1279 | 92.9 | 93.9 |
    | 3DLoMatch | 1726 | 71.9 | 74.5 | 

- Recall on 3DMatch and 3DLoMatch (under 0.3m && 15 degrees)

    | Dataset | npairs | Pair Recall (%) |
    | :---: | :---: | :---: |
    | 3DMatch | 1623 | 95.0 |
    | 3DLoMatch | 1781 | 75.1 | 

- Results on Odometry KITTI

    | Dataset | RTE(cm) | RRE(°) | Recall (%) |
    | :---: | :---: | :---: | :---: |
    | Odometry KITTI | 6.1 | 0.26 | 99.8 |

- Results on MVP-RG

    | Dataset | RRE(°) | RTE | RMSE |
    | :---: | :---: | :---: | :---: |
    | MVP-RG | 7.99 | 0.048 | 0.093 |

## Environments

- All experiments were run on a RTX 3090 GPU with an  Intel 8255C CPU at 2.50GHz CPU.  Dependencies can be found in `requirements.txt`.

- Compile python bindings

    ```
    # Compile

    cd NgeNet/cpp_wrappers
    sh compile_wrappers.sh
    ```

## [Pretrained weights (Optional)]

Download pretrained weights for 3DMatch, 3DLoMatch, Odometry KITTI and MVP-RG from [GoogleDrive](https://drive.google.com/drive/folders/1JDn6zQfLdZfAVVboXRrrrCVRo48pRjyW?usp=sharing) or [BaiduDisk](https://pan.baidu.com/s/18G_Deim1UlSkY8wWoOiwnw) (pwd: `vr9g`).

## [3DMatch and 3DLoMatch]

### 1.1 dataset

We adopt the 3DMatch and 3DLoMatch provided from [PREDATOR](https://github.com/overlappredator/OverlapPredator), and download it [here](https://share.phys.ethz.ch/~gsg/Predator/data.zip) [**936.1MB**].
Unzip it, then we should get the following directories structure:

``` 
| -- indoor
    | -- train (#82, cats: #54)
        | -- 7-scenes-chess
        | -- 7-scenes-fire
        | -- ...
        | -- sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika_4
    | -- test (#8, cats: #8)
        | -- 7-scenes-redkitchen
        | -- sun3d-home_md-home_md_scan9_2012_sep_30
        | -- ...
        | -- sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
```

### 1.2 train

```
## Reconfigure configs/threedmatch.yaml by updating the following values based on your dataset.

# exp_dir: your_saved_path for checkpoints and summary.
# root: your_data_path for the 3dMatch dataset.

cd NgeNet
python train.py configs/threedmatch.yaml

# note: The code `torch.cuda.empty_cache()` in `train.py` has some impact on the training speed.
# You can remove it or change its postion according to your GPU memory. 
```

### 1.3 evaluate and visualize

```
cd NgeNet

python eval_3dmatch.py --benchmark 3DMatch --data_root your_path/indoor --checkpoint your_path/3dmatch.pth --saved_path work_dirs/3dmatch [--vis] [--no_cuda]

python eval_3dmatch.py --benchmark 3DLoMatch --data_root your_path/indoor --checkpoint your_path/3dmatch.pth --saved_path work_dirs/3dlomatch [--vis] [--no_cuda]
```

## [Odometry KITTI]

### 2.1 dataset

Download odometry kitti [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) with `[velodyne laser data, 80 GB]` and `[ground truth poses (4 MB)]`, then unzip and organize in the following format.

```
| -- kitti
    | -- dataset
        | -- poses (#11 txt)
        | -- sequences (#11 / #22)
    | -- icp (generated automatically when training and testing)
        | -- 0_0_11.npy
        | -- ...
        | -- 9_992_1004.npy
```

### 2.2 train

```
## Reconfigure configs/kitti.yaml by updating the following values based on your dataset.

# exp_dir: your_saved_path for checkpoints and summary.
# root: your_data_path for the Odometry KITTI.

cd NgeNet
python train.py configs/kitti.yaml
```

### 2.3 evaluate and visualize

```
cd NgeNet
python eval_kitti.py --data_root your_path/kitti --checkpoint your_path/kitti.pth [--vis] [--no_cuda]
```

## [MVP-RG]

### 3.1 dataset

Download MVP-RG dataset [here](https://mvp-dataset.github.io/MVP/Registration.html), then organize in the following format.

```
| -- mvp_rg
    | -- MVP_Train_RG.h5
    | -- MVP_Test_RG.h5
```

### 3.2 train

```
## Reconfigure configs/mvp_rg.yaml by updating the following values based on your dataset.

# exp_dir: your_saved_path for checkpoints and summary.
# root: your_data_path for the MVP-RG.

python train.py configs/mvp_rg.yaml

# note: The code `torch.cuda.empty_cache()` in `train.py` has some impact on the training speed.
# You can remove it or change its postion according to your GPU memory. 
```

### 3.3 evaluate and visualize

```
python eval_mvp_rg.py --data_root your_path/mvp_rg --checkpoint your_path/mvp_rg.pth [--vis] [--no_cuda]
```

## [Demo]

### 4.1 3DMatch

```
python demo.py --src_path demo_data/cloud_bin_21.pth --tgt_path demo_data/cloud_bin_34.pth --checkpoint your_path/3dmatch.pth --voxel_size 0.025 --npts 5000
```
![](demo_data/3dmatch.png)

### 4.2 Personal data (with the same voxel size as 3DMatch)

```
python demo.py --src_path demo_data/src1.ply --tgt_path demo_data/tgt1.ply --checkpoint your_path/3dmatch.pth  --voxel_size 0.025 --npts 20000
```
![](demo_data/my_data1.png)

### 4.3 Personal data (with different voxel size from 3DMatch)

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
