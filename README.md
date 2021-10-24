# [Point Cloud Registration using Representative Overlapping Points (ROPNet)](https://arxiv.org/abs/2107.02583)

- Our solution based on ROPNet and [OverlapPredator](https://github.com/overlappredator/OverlapPredator) won **the second place** on the [MVP Registration Challenge (ICCV Workshop 2021)](https://mvp-dataset.github.io/MVP/Registration.html). [[Technical Report](https://arxiv.org/abs/2110.09129)], [[Zhihu](https://zhuanlan.zhihu.com/p/423011401)]

## Abstract

3D point cloud registration is a fundamental task in robotics and computer vision. Recently, many learning-based point cloud registration methods based on correspondences have emerged. However, these methods heavily rely on such correspondences and meet great challenges with partial overlap. In this paper, we propose ROPNet, a new deep learning model using Representative Overlapping Points with discriminative features for registration that transforms partial-to-partial registration into partial-to-complete registration. Specifically, we propose a context-guided module which uses an encoder to extract global features for predicting point overlap score. To better find representative overlapping points, we use the extracted global features for coarse alignment. Then, we introduce a Transformer to enrich point features and remove non-representative points based on point overlap score and feature matching. A similarity matrix is built in a partial-to-complete mode, and finally, weighted SVD is adopted to estimate a transformation matrix. Extensive experiments over ModelNet40 using noisy and partially overlapping point clouds show that the proposed method outperforms traditional and learning-based methods, achieving state-of-the-art performance.

## Environment

The code has been tested on Ubuntu 16.04, Python 3.7, PyTorch 1.7, Open3D 0.9.

## Dataset

Download [ModelNet40](https://modelnet.cs.princeton.edu) from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) [435M].

## Model Training

```
cd src/
python train.py --root your_data_path/modelnet40_ply_hdf5_2048/ --noise --unseen
```

## Model Evaluation

```
cd src/
python eval.py --root your_data_path/modelnet40_ply_hdf5_2048/  --unseen --noise  --cuda --checkpoint work_dirs/models/min_rot_error.pth
```

## Registration Visualization

```
cd src/
python vis.py --root your_data_path/modelnet40_ply_hdf5_2048/  --unseen --noise  --checkpoint work_dirs/models/min_rot_error.pth
```


## Citation

If you find our work is useful, please consider citing:

```
@article{zhu2021point,
  title={Point Cloud Registration using Representative Overlapping Points},
  author={Zhu, Lifa and Liu, Dongrui and Lin, Changwei and Yan, Rui and G{\'o}mez-Fern{\'a}ndez, Francisco and Yang, Ninghua and Feng, Ziyong},
  journal={arXiv preprint arXiv:2107.02583},
  year={2021}
}
```

and

```
@article{zhu2021deep,
  title={Deep Models with Fusion Strategies for MVP Point Cloud Registration},
  author={Zhu, Lifa and Lin, Changwei and Liu, Dongrui and Li, Xin and G{\'o}mez-Fern{\'a}ndez, Francisco},
  journal={arXiv preprint arXiv:2110.09129},
  year={2021}
}
```

## Acknowledgements

We thank the authors of [RPMNet](https://github.com/yewzijian/RPMNet), [PCRNet](https://github.com/vinits5/pcrnet_pytorch), [OverlapPredator](https://github.com/overlappredator/OverlapPredator), [PCT](https://github.com/MenghaoGuo/PCT) and [PointNet++](https://github.com/charlesq34/pointnet2) for open sourcing their methods.

We also thank the third-party code [PCReg.PyTorch](https://github.com/zhulf0804/PCReg.PyTorch) and [Pointnet2.PyTorch](https://github.com/zhulf0804/Pointnet2.PyTorch).