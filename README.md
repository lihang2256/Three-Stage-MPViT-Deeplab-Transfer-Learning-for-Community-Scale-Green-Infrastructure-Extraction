# Three-Stage-MPViT-Deeplab-Transfer-Learning-for-Community-Scale-Green-Infrastructure-Extraction
## 1 Introduction
This repository is the code and data of paper Three-Stage-MPViT-Deeplab-Transfer-Learning-for-Community-Scale-Green-Infrastructure-Extraction. In this paper, we reannotate a training dataset of CSGI and propose a three-stage transfer learning method employing a novel hybrid architecture, MPViT-Deeplab to help us focus on the training task and improve accuracy. In MPViT-Deeplab, Multi-path Vision Transformer (MPViT) serves as the feature extractor, feeding both coarse and fine features into the decoder and encoder of Deeplabv3+ respectively, which enables pixel-level segmentation of CSGI in remote sensing images. 
## 2 Data Source
In this experiment, we use a few datasets and reannotated DroneDeploy Segmentation Dataset. We list data sources below:
- CityScapes: [https://www.cityscapes-dataset.com/]
- imagenet2012: [https://image-net.org/challenges/LSVRC/2012/index.php]
- Land/land1: [https://paperswithcode.com/dataset/second]
- Land/land2-4: [https://github.com/dronedeploy/dd-ml-segmentation-benchmark]
- csgi(our reannotated data in Datasets/data/relabel): [链接：https://pan.baidu.com/s/1H9hduodl94KrutBIG0xq9g 提取码：knqr]
## 3 Run
Training environment is in environment.txt.

Training parameters is in train-parm.txt.

Download data, configure the environment, set parameters and run main.py.
## 4 Contact Us
If you are interested in our paper or face any problems, please email us by 371615606@qq.com.


