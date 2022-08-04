# Face Recongnition with Yolov5 + StrongSORT with OSNet and ResNet50 FR classifier

## Introduction

This repository contains a highly configurable two-stage-tracker that adjusts to different deployment scenarios. The detections generated by [YOLOv5](https://github.com/ultralytics/yolov5), a family of object detection architectures and models pretrained on the COCO dataset, are passed to [StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/pdf/2202.13514.pdf) which combines motion and appearance information based on [OSNet](https://github.com/KaiyangZhou/deep-person-reid)[](https://arxiv.org/abs/1905.00953) in order to tracks the objects. It can track any object that your Yolov5 model was trained to detect.



## Before you run the tracker

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

`pip install -r requirements.txt`


## Tracking sources

Tracking can be run on most video formats

```bash
$ docker pull nitinroxx/nvidia-tensorflow-11.6:v1
$ sudo nvidia-docker run -it --net=host -v `pwd`:/WS/ --entrypoint bash nitinroxx/nvidia-tensorflow-11.6:v1
$ root@68dc4e0c6db8:/WS# ./facesense.sh
```


## Contact 

For Yolov5 DeepSort OSNet bugs and feature requests please visit [GitHub Issues](https://github.com/Nitin286roxs/Facesense/issues). For business inquiries or professional support requests please send an email to: nitinashu1994@gmail.com
