# Face Recongnition with Yolov5 + StrongSORT with OSNet and ResNet50 FR classifier

![avatar](demo.gif)

## Introduction

This repository contains a highly configurable two-stage-tracker that adjusts to different deployment scenarios. The detections generated by [YOLOv5](https://github.com/ultralytics/yolov5), a family of object detection architectures and models pretrained on the COCO dataset, are passed to [StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/pdf/2202.13514.pdf) which combines motion and appearance information based on [OSNet](https://github.com/KaiyangZhou/deep-person-reid)[](https://arxiv.org/abs/1905.00953) in order to tracks the objects. It can track any object that your Yolov5 model was trained to detect.



## Before you run the tracker

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Download yolov5 model weight, trained classifier weights and test video from [Drive](https://drive.google.com/drive/folders/1L42bVQXbcNH2kV56Ep9otcry2pDB1hVQ?usp=sharing)

3. Run mongodb and apache flask server same docker network
```bash
$ sudo docker network create -d bridge facesense-bridge
$ sudo docker run -d -v /home/nitin/Safepro/Workspace/CrowdAttendence/Yolov5_StrongSORT_OSNet/facesense_db:/data/db -p 27017:27017  --network="facesense-bridge" --name mongodb  mongo:latest
$ sudo docker run -d -p 80:80 --name api_server -v /home/nitin/Safepro/Workspace/CrowdAttendence/facesense_endpoints/docker_compose_for_apache_flask_setup-main/apache-flask-master/routes.py:/var/www/apache-flask/app/routes.py --network="facesense-bridge" apache-flask-server:v1
```

3. Pull prebuilt docker image, and run the facesense.sh shell file

```bash
$ docker pull nitinroxx/facesense:v1
$ sudo nvidia-docker run -it --net=host --name facesense_local_testing -v `pwd`:/WS/ --entrypoint bash nitinroxx/facesense:v1
$ root@68dc4e0c6db8:/WS# ./facesense.sh
```


## Contact 

For Yolov5 DeepSort OSNet bugs and feature requests please visit [GitHub Issues](https://github.com/Nitin286roxs/Facesense/issues). For business inquiries or professional support requests please send an email to: nitinashu1995@gmail.com
