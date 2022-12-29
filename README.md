# Introduction
Released code for the [paper](https://arxiv.org/abs/2204.08717): <i>a single-stage monocular 3D object detection model is proposed. An instance-segmentation head is integrated into the model training, which allows the model to be aware of the visible shape of a target object. </i>

# Installation

The code is tested with 
`python==3.7, torch==1.4.0, torchvision==0.5.0`.

After python and pytorch are installed on your machine, You can run the following commands to prepare the environment:
```
pip install -r requirements.txt 
cd mmdet3d/ops/DCNv2
. make.sh 
cd ../../..
pip install -v -e . 
```

# Data Preparation 
Please download the [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the data as follows 
```
data/kitti/     
        |training/
            |calib/
            |image_2/
            |label/
            |ImageSets/
        |testing/
            |calib/
            |image_2/
            |ImageSets/
```

# Training & Evaluation 

training:
```
python -m torch.distributed.launch --nproc_per_node=<number of gpus> mono3d/train.py <config file> --no-validate --launcher=pytorch
```

evaluation:
```
python mono3d/test.py <config file> <checkpoint file> --eval bbox 
```

# Model Zoo
|Config File|  Model|AP(3D)|
|-|-|-|
|configs_shapeaware/shapeaware_quantizationFloor_albu_SAFLw2NoRepair_CS.py|[Google Drive](https://drive.google.com/file/d/1eB-Xdm9LAsXQCovL-QUQHCGwNkxjw4EV/view?usp=share_link)|18.39|

# Acknowledgements
The project benefits a lot from the following works. Thanks for their contribution. 
- [MonoFlex]()
- [MMDetection3D]()
