# Source Code of G-Rep

## Introduction

The source code includes training and inference procedures for the proposed method of the paper “G-Rep: Gaussian Representation for Arbitrary-Oriented Object Detection” https://doi.org/10.3390/rs15030757 (doi: 10.3390/rs15030757)

This part of the code illustrates the effectiveness of the proposed method choosing [RepPoints](https://ieeexplore.ieee.org/document/9009032) as the baseline. The implementation of the baseline method for comparison, RepPoints with oriented bounding boxes, comes from [BeyondBoundingBox](https://github.com/SDL-GuoZonghao/BeyondBoundingBox), which also adopts [MMDetection](https://github.com/open-mmlab/mmdetection) framework.

It is recommended to get the MMDetection framework and some necessary CUDA functions  by installing [BeyondBoundingBox](https://github.com/SDL-GuoZonghao/BeyondBoundingBox) directly and copy the  source code of the proposed method  to its source code tree for usage.

## Dependencies

- Python 3.5+ (Python 2 is not supported)
- PyTorch 1.2
- CUDA 9.0+
- NCCL 2
- GCC (G++) 4.9+
- [mmdetection](https://github.com/open-mmlab/mmdetection) 1.1.0 
- [mmcv](https://github.com/open-mmlab/mmcv) 0.2.14
- [BeyondBoundingBox](https://github.com/SDL-GuoZonghao/BeyondBoundingBox)


We have tested the code on the following versions of OS and softwares:

- OS:  Ubuntu 16.04 LTS
- Python: 3.7 (installed along with Anaconda 3)
- CUDA: 10.1
- NCCL: 2.3.7
- GCC (G++): 7.5
- PyTorch: 1.2

## Instructions for Usage
### Setp1: Create Environment
a. Create a virtual environment and activate it in Anaconda:

```bash
conda create -n reppoints python=3.7 -y
conda activate reppoints
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/):

```bash
conda install pytorch=1.2 torchvision cudatoolkit=10.0 -c pytorch
```
### Setp2: Install BeyondBoundingBox with MMDetection
a. Clone BeyondBoundingBox to current path.
```bash
git clone https://github.com/SDL-GuoZonghao/BeyondBoundingBox.git
cd BeyondBoundingBox
```
b. Install other dependencies and setup BeyondBoundingBox.
```bash
pip install -r requirements.txt
python setup.py develop
```

### Setp3: Prepare datasets
It is recommended to make a symbolic link of the dataset root to ``data`` and install [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) in BeyondBoundingBox root path.

The following instructions are for converting annotations of each dataset to the format of MMDetection. They are provided just for reference. You can achieve the same goal in whatever way convenient to you.

**Note**: the corresponding dataset path and class name should be changed in the python scripts according to actual position and setting of the data, and current path for executing the instructions is ``BeyondBoundingBox``.

- **DOTA**

  Please refer to [DOTA](https://captain-whu.github.io/DOTA/index.html) to get the training, validation and test set.
  Before training, the image-splitting process must be carried out.  See [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) for details. 
```
python ./DOTA_devkit/DOTA2COCO.py 
```

- **HRSC2016**
	Please refer to [HRSC2016](https://sites.google.com/site/hrsc2016/) to get the training, validation and test set.
```bash
mv ../data_prepare/hrsc2016/HRSC2DOTA.py ./DOTA_devkit/
mv ../data_prepare/hrsc2016/HRSC2JSON.py ./DOTA_devkit/	
mv ../data_prepare/hrsc2016/prepare_hrsc2016.py ./DOTA_devkit/
python ./DOTA_devkit/prepare_hrsc2016.py 
```


-  **UCAS-AOD**
	Please refer to [UCAS-AOD](https://hyper.ai/datasets/5419 ) to get the training, validation and test set.
	Please run the following scripts in sequence, and note that the corresponding dataset paths should be changed in these scripts accordingly:
```bash
python ../data_prepare/ucas-aod/data_prepare.py
python ../data_prepare/ucas-aod/prepare_ucas.py
python ./DOTA_devkit/DOTA2COCO.py
```
- **ICDAR2015**
	Please refer to [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) to get the training, validation and test set. 
```bash
python ../data_prepare/icdar2015/prepare_ic15.py
python ./DOTA_devkit/DOTA2COCO.py
```

### Setp4: Install the source codes of G-Rep

**(a)** Add source files of the proposed method to BeyondBoundingBox (execute following commands in the sub-directory ``BeyondBoundingBox``)

```shell
 mv ../icdar2015.py ./mmdet/datasets/
 mv ../ucas_aod.py ./mmdet/datasets/
 
 mv ../ucas_aod.py ./mmdet/models/detectors/
 
 mv ../kld_loss.py ./mmdet/models/losses/
 mv ../bhattacharyya_loss.py ./mmdet/models/losses/
 mv ../wtd_loss.py ./mmdet/models/losses/
 
 mv ../rreppoints_head.py ./mmdet/models/anchor_heads/
 mv ../grep_reppoints_head.py ./mmdet/models/anchor_heads/
 mv ../atss_reppoints_head.py ./mmdet/models/anchor_heads/
 mv ../grep_reassign_head.py ./mmdet/models/anchor_heads/
 
 mv ../transforms_kld.py ./mmdet/core/bbox/
 mv ../max_convex_iou_assigner.py ./mmdet/core/bbox/assigners/
 mv ../max_gaussian_iou_assigner.py ./mmdet/core/bbox/assigners/
 mv ../atss_convex_assigner.py ./mmdet/core/bbox/assigners/
 mv ../atss_kld_assigner.py ./mmdet/core/bbox/assigners/
 mv ../max_kld_assigner.py ./mmdet/core/bbox/assigners/
 mv ../max_bcd_assigner.py ./mmdet/core/bbox/assigners/
 mv ../atss_bcd_assigner.py ./mmdet/core/bbox/assigners/
 mv ../max_wtd_assigner.py ./mmdet/core/bbox/assigners/
 mv ../atss_wtd_assigner.py ./mmdet/core/bbox/assigners/
 
 mv ../gmm.py ./mmdet/ops/
 mv -r ../point_justify ./mmdet/ops/
 mv -r ../torch_batch_svd ./mmdet/ops/
 rm ./setup.py
 mv ../setup.py ./

 mv ../dota_beyond_read_pkl.py ./tools/
 mv -r ../data_prepare/icdar2015/icdar2015_evalution ./tools/ # see readme.txt for detail
 mv ../data_prepare/hrsc2016/hrsc2016_evaluation.py ./DOTA_devkit/
 mv ../data_prepare/ucas-aod/ucas_aod_evaluation.py ./DOTA_devkit/
```

**(b)** Import new core classes and functions in \__init__.py 

- Import core classes and functions in ``/mmdet/models/detectors/__init__.py``
```shell
from .rreppoints_detector import RRepPointsDetector

__all__ = ['RRepPointsDetector']
```

- Import core classes and functions in ``./mmdet/core/bbox/assigners/__init__.py`` 
```shell
from .max_convex_iou_assigner import MaxConvexIoUAssigner
from .atss_convex_assigner import ATSSConvexAssigner
from .max_gaussian_iou_assigner import MaxGaussianIoUAssigner
from .max_kld_assigner import MaxKLDAssigner
from .atss_kld_assigner import ATSSKldAssigner
from .atss_bcd_assigner import ATSSBcdAssigner
from .atss_wtd_assigner import ATSSWtdAssigner
from .max_bcd_assigner import MaxBcdAssigner
from .max_wtd_assigner import MaxWtdAssigner

__all__ = ['MaxConvexIoUAssigner', 'ATSSConvexAssigner', 'MaxKLDAssigner'
    , 'ATSSKldAssigner', 'MaxGaussianIoUAssigner', 'ATSSBcdAssigner', 'MaxBcdAssigner'
    , 'ATSSWtdAssigner', 'MaxWtdAssigner']
```

- Import core classes and functions to ``./mmdet/models/anchor_heads/__init__.py`` 
```shell
from .grep_reppoints_head import GRepRepPointsHead
from .grep_reassign_head import GRepReassignHead
from .atss_reppoints_head import ATSSRepPointsHead
from .rreppoints_head import RRepPointsHead

__all__ = ['RRepPointsHead', 'ATSSRepPointsHead', 'GRepRepPointsHead', 'GRepReassignHead']
```

- Import core classes and functions to ``./mmdet/models/losses/__init__.py`` 
```shell
from .kld_loss import KLDLoss
from .bhattacharyya_loss import BhattacharyyaLoss
from .wtd_loss import WassersteinLoss
__all__ = ['KLDLoss', 'BhattacharyyaLoss', 'WassersteinLoss']
```

**(c)** Installation of additional cuda functions 

```
cd ./mmdet/ops/point_justify/
python setup.py install
python setup.py build_ext  --inplace
```

### Setp5: Prepare config files

For each dataset, we provide sample configures for both the baseline and our method in the files under the sub-directory ``configs``. For example, ``pointset_atssiou_giou_r50_1x_dota.py`` provides the configures for the proposed method running on DOTA dataset.

### Setp6: Train and test
Take the same instructions as BeyondBoundingBox. See [Training and Inference](https://github.com/SDL-GuoZonghao/BeyondBoundingBox) for details.



## Citation

If this is useful for your research, please consider citing following works.

```
@article{hou2023g,
  title={G-rep: Gaussian representation for arbitrary-oriented object detection},
  author={Hou, Liping and Lu, Ke and Yang, Xue and Li, Yuqiu and Xue, Jian},
  journal={Remote Sensing},
  volume={15},
  number={3},
  pages={757},
  year={2023},
  publisher={MDPI}
}

@inproceedings{guo2021beyond,
  title={Beyond Bounding-Box: Convex-Hull Feature Adaptation for Oriented and Densely Packed Object Detection},
  author={Guo, Zonghao and Liu, Chang and Zhang, Xiaosong and Jiao, Jianbin and Ji, Xiangyang and Ye, Qixiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8792--8801},
  year={2021}
}

@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}

@article{chen2019mmdetection,
  title={MMDetection: Open mmlab detection toolbox and benchmark},
  author={Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and others},
  journal={arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
