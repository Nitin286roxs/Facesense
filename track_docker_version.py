#Hall ROI
#polygon_roi = [[640,275], [1345,275], [1700,1080], [455,1080]] #OLD View

# Gallery roi
polygon_roi = [[0,200], [1340,200], [1340,1080], [0,1080]]
polygon_roi = [[350,405], [2304,405], [2304,1296], [350,1296]]

#polygon_roi = [[574,338], [1704,338], [1750, 520], [1528,1295], [665,1295]] #Final ROI
import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path
import random
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'
print(f"ROOT: {ROOT}")

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
#setting path for gaze estimation
if str(ROOT / 'gaze_estimation') not in sys.path:
    sys.path.append(str(ROOT / 'gaze_estimation'))  # add strong_sort ROOT to PA
if str(ROOT / 'deepface') not in sys.path:
    sys.path.append(str(ROOT / 'deepface'))  # add strong_sort ROOT to PATH
if str(ROOT / 'face_recognition_embedding/') not in sys.path:
    sys.path.append(str(ROOT / 'face_recognition_embedding/'))  # add strong_sort ROOT to PATH
print(f"ROOT: {ROOT}")
print(f"sys.path.: {sys.path}")
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print(f"ROOT: {ROOT}")
import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from tensorflow.keras.utils import img_to_array
from gaze_estimation import get_eye_box
#from keras.models import load_model
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Convolution2D, LocallyConnected2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
from timm.models import create_model
from resnet50_feature_extract import load_trainable_model, getResNet50Model
import torchvision.transforms as transforms
import face_recognition
from deepface import DeepFace
import base64
from pymongo import MongoClient
# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

def automate_histogram_mid(mean):
    # For Darker Image
    if mean < 75:
        mid = random.choice([0.5, 0.6, 0.7])
    # For Average Lighting Image
    elif mean >= 75 and mean <= 130:
        mid = random.choice([.2,.3,.4, 0.5, 0.6, 0.7])
    # For Brighter Image
    else:
        mid = random.choice([0.2, 0.3, 0.4,.5])
    return mid

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    # apply gamma correction using the lookup table
    return cv2.LUT(image, lookUpTable)

def draw_mask(cv_img):
    from skimage import filters
    
    #img_path = sys.argv[1]
    #cv_img = cv2.imread(img_path)
    print(cv_img.shape)
    mask = np.zeros_like(cv_img)
    # fill polygon
    poly = np.array([[
        (1370, 0),
        (1610, 0),
        (1920, 140),
        (1920, 980),
        (1360, 230),
    ]], dtype = np.int32)
    factor = 1.0
    k_w = int(cv_img.shape[1] * factor*0.1)
    k_h = int(cv_img.shape[0] * factor*0.1)
    # ensure the width of the kernel is odd
    if k_w % 2 == 0:
        k_w -= 1
    # ensure the height of the kernel is odd
    if k_h % 2 == 0:
        k_h -= 1
    # apply a Gaussian blur to the input image using our computed
    # kernel size
    blurred_image = cv2.GaussianBlur(cv_img,(abs(k_w), abs(k_h)), 0)
    mask = np.zeros(cv_img.shape, dtype=np.uint8)
    channel_count = cv_img.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, poly, ignore_mask_color)
    mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
    final_image = cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(cv_img, mask_inverse)
    return final_image

#Intersection over union check for two objects
def iou_check(box_obj, box_roi):
    from shapely.geometry import Polygon
    box_obj = [(box_obj[0], box_obj[1]),(box_obj[2],box_obj[1]),(box_obj[2],box_obj[3]), (box_obj[0], box_obj[3])]
    polygon1 = Polygon(box_obj)
    polygon2 = Polygon(box_roi)
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.area + polygon2.area - intersect #polygon1.union(polygon2).area
    iou = intersect / polygon1.area
    return iou  # iou = 0.5

def l2_normalize(x):
	return x / np.sqrt(np.sum(np.multiply(x, x)))

def findEuclideanDistance(source_representation, test_representation):
	euclidean_distance = source_representation - test_representation
	euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
	euclidean_distance = np.sqrt(euclidean_distance)
	return euclidean_distance

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load face classifier model
    # create model
    #face_model_type = "vggnet" #"resnet50"
    face_model_type = "embeded_FR"
    model_checkpoints = "resnet50_face_classifier/checkpoint-19.pth.tar" #checkpoint-32.pth.tar"
    number_of_person = 8 #9
    face_name = ["GOWTHAMI", "KARTHIK", "KIRAN", "MANIKCHAND", "NITIN", "PAVAN", "RAJU", "RAKSHITA", "SANTHA", "SONU", "SRIDHAR", "SUJAY", "VIDYA"]
    if face_model_type == "resnet50":
        face_model = create_model(
            face_model_type,
            num_classes=number_of_person,
            in_chans=3,
            pretrained=False,
            checkpoint_path=model_checkpoints)
        face_model = face_model.cuda()
        face_model.eval()
    # Imagezmq Sender
    import imagezmq
    sender = imagezmq.ImageSender(connect_to='tcp://*:5555', REQ_REP=False)
    host_name = 'From Sender'
    # OpenCV facedetection model 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Dataloader
    if webcam:
        show_vid = False #check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(polygon_roi, source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(polygon_roi, source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    gender_model_path = "face_classification.model"
    gender_model = load_model(gender_model_path)
    gender_classes = ["GOWTHAMI", "NITIN", "PAVAN", "RAJU", "SANTHA", "SONU", "SRIDHAR", "SUJAY", "VIDYA"]
    gender_label = "Unknown"
    unique_seen_count_bilboard = {"male":[],"female":[]}
    #TODO record attendence and trackID
    person_attendence = []
    #attendence_record = []
    curr_date = None
    curr_time = None
    is_csv_dumped = False
    face_buffer = {}
    curr_tracked_id = []
    imgstring_buffer = {}
    field_names = ['EmpoloyeeName', 'Availibilty', 'AttendenceTime', "EntranceId"]
    #Loading Embeded classifier
    face_model_type = "embeded_FR"
    employees = dict()
    known_people_folder = None
    buffer_length = 2 
    id_buffer_len = 80
    if face_model_type == "embeded_classifier":
        # Get similar images of test images for ResNet-50 (b)
        resnet_model_b = getResNet50Model(lastFourTrainable=True)
        resnet_model_b.load_weights('./model_resnet_trainable.h5')
        feature_model_resnet_b = Model(inputs=resnet_model_b.input, outputs=resnet_model_b.get_layer('new_fc').output)

        df = pd.read_pickle('./features_resnet_b.pickle')
        #for file in feature_test_files:
    elif face_model_type == "embeded_FR":
        known_people_folder = "./face_recognition_embedding/data"
        known_face_encodings = []
        known_names = []
        for person_name in face_name:
            img_to_read = f"{known_people_folder}/{person_name}.jpg"
            img_path = None
            if os.path.isfile(f"{known_people_folder}/{person_name}.jpg"):
                img_path = img_to_read
            elif os.path.isfile(f"{known_people_folder}/{person_name}.jpeg"):
                img_path = f"{known_people_folder}/{person_name}.jpeg"
            else:
                img_path =  f"{known_people_folder}/{person_name}.png"
            print(f"img_path: {img_path}")
            if os.path.isfile(img_path):
                person_image = face_recognition.load_image_file(img_path) 
                person_face_encoding = face_recognition.face_encodings(person_image)[0]
                known_face_encodings.append(person_face_encoding)
                known_names.append(person_name)
        #known_names, known_face_encodings = scan_known_people(known_people_folder)
    for person_name in face_name:
        temp_attendence = {}
        temp_attendence["EmpoloyeeName"] = person_name
        temp_attendence["Availibilty"] = "Absent"
        temp_attendence["AttendenceTime"] = None
        temp_attendence["EntranceId"] = []
        person_attendence.append(temp_attendence)
    print(f"person_attendence: {person_attendence}")
    # MongoDB 
    client = MongoClient("mongodb://safepro:facesense321@192.168.0.156:27017/attendance")
    #client = MongoClient("mongodb://safepro:facesense321@3.7.179.66:27017/attendance")
    #client = MongoClient('mongodb://localhost:27017/')
    db = client["attendance"]
    daily = db.daily

    last_tracked_id = []
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Running openvino model for face and eye dertection
        isFace = False
        #s_time = time.time()*1000
        is_face_visible = True
        #im0s, is_face_visible = get_eye_box([], im0s, isFace, polygon_roi)
        #e_time = time.time()*1000
        dt[2] += time_sync() - t3
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0
            #Draw ROI
            pts = np.array(polygon_roi,np.int32)
            pts = pts.reshape((-1, 1, 2))
            isClosed = True
            color = (0, 255, 0)
            thickness = 2
            #Drawing blurring on mirror
            s_time = time.time()*1000
            #im0 = draw_mask(im0)
            e_time = time.time()*1000
            import copy
            deep_im0 = copy.deepcopy(im0)
            print(f"Time taken to draw mask: {e_time - s_time}msec.")
            im0 = cv2.polylines(im0, [pts],
                      isClosed, color, thickness)
            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            current_tracked_id = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = int(output[4])
                        current_tracked_id.append(id)
                        cls = output[5]
                        roi = [polygon_roi[0][0], polygon_roi[0][1],\
                                                     polygon_roi[1][0], polygon_roi[2][1]]
                        iou_conf = iou_check(bboxes, polygon_roi)
                        if iou_conf > 0.8 and int(cls)==1:
                            gender_label = "Unknown"
                            curr_tracked_id.append(id)
                            face_gamma = None
                            person_crop = deep_im0[int(output[1]):int(output[3]), int(output[0]):int(output[2])]
                            # apply face classifier on face
                            if face_model_type == "vggnet": 
                                face_crop = cv2.resize(person_crop, (112,112))
                                face_crop = face_crop.astype("float") / 255.0
                                face_crop = img_to_array(face_crop)
                                face_crop = np.expand_dims(face_crop, axis=0)
                                gender_conf = gender_model.predict(face_crop)[0]
                                # get label with max accuracy
                                idx = np.argmax(gender_conf)
                                gender_label = gender_classes[idx]
                                #age_gender_label = f"{id} {label}: {conf:.2f}"
                            elif face_model_type == "full_classifier":
                                # Resnet50 preprocessing
                                s_time = time.time()*1000
                                face_crop = cv2.resize(person_crop, (112,112))
                                resnet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                                resnet_scale = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                face_crop_rgb = face_crop_rgb.astype("float") / 255.0
                                face_crop_rgb = (face_crop_rgb - resnet_mean) / resnet_scale
                                face_crop_tensor = face_crop_rgb.transpose((2, 0, 1))
                                face_crop_cuda = np.ndarray(shape=(1, 3, 112, 112),dtype=float)
                                face_crop_cuda[0] = face_crop_tensor
                                face_crop_cuda = torch.as_tensor(face_crop_cuda,device=torch.device('cuda'), dtype=torch.float)
                                e_time = time.time()*1000
                                print(f"ResNet50 preprocessing: {e_time-s_time} msec.")
                                s_time = time.time()*1000
                                labels = face_model(face_crop_cuda)
                                e_time = time.time()*1000
                                print(f"ResNet50 classifier infertime: {e_time-s_time} msec.")
                                topk = labels.topk(3)[1].cpu().numpy()[0][0]
                                gender_label = face_name[int(topk)]
                            elif face_model_type == "embeded_FR":
                                #TODO Face gamma correction
                                if is_face_visible:
                                    face_temp_gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                                    mean = np.mean(face_temp_gray)
                                    mid = automate_histogram_mid(mean)
                                    gamma = math.log(mid)/math.log(mean/255)
                                    # do gamma correction
                                    face_gamma = adjust_gamma(person_crop, gamma)
                                    s_time = time.time()*1000
                                    #small_frame = cv2.resize(person_crop, (0, 0), fx=0.25, fy=0.25)
                                    unknown_image = cv2.cvtColor(face_gamma, cv2.COLOR_BGR2RGB)
                                    pil_image=Image.fromarray(unknown_image)
                                    se_time = time.time()*1000
                                    face_locations = face_recognition.face_locations(unknown_image)
                                    ee_time = time.time()*1000
                                    print(f"time to get face location: {ee_time-se_time }msec")
                                    print(f"len of face bbox: {int(output[1]), int(output[3]), int(output[0]), int(output[2])}")
                                    print(f"len of face locations: {(face_locations)}")
                                    if len(face_locations):
                                        se_time = time.time()*1000
                                        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                                        ee_time = time.time()*1000
                                        print(f"time to get face embedding: {ee_time-se_time }msec")
                                        print(f"len of face encodings: {len(face_encodings)}")
                                        matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                                        name = "Unknown"
                                        face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
                                        best_match_index = np.argmin(face_distances)
                                        if matches[best_match_index]:
                                            gender_label = face_name[best_match_index]
                                        else:
                                            gender_label = name
                                    e_time = time.time()*1000
                                    print(f"time taken to FR: {e_time-s_time}msec.")
                                else:
                                    gender_label = "Unknown"
                            elif face_model_type == "deepface":
                                cv2.imwrite("temp_face.jpg", person_crop)
                                df_res = DeepFace.find(img_path = "temp_face.jpg",\
                                                       db_path = "./face_recognition_embedding/data",\
                                                       model_name="Ensemble",\
                                                       enforce_detection=False)
                                print(f"df_res: {df_res}")
                                if not df_res.empty:
                                    gender_label = (df_res.head(1)['identity'].values[0].split("/")[-1][0:-4])
                                print(f"gender_label: {gender_label}")
                                
                            else:
                                sim_conf, top_id = load_trainable_model(person_crop,df, feature_model_resnet_b)
                                print(f"sim_con: {sim_conf}, top_id: {top_id}!!")
                                gender_label = face_name[int(top_id)]
                            #Track all faces
                            if str(id) not in face_buffer:
                                face_buffer[str(id)] = {}
                            print(f"gender_label: {gender_label}")
                            if gender_label!="Unknown":
                                if gender_label not in face_buffer[str(id)]:
                                    face_buffer[str(id)][gender_label] = 0
                                face_buffer[str(id)][gender_label] += 1 
                                #Check if buffer length is 20
                                if len(curr_tracked_id)>=id_buffer_len:
                                    curr_tracked_id.pop(0)
                                unique_ids = sorted(list(set(curr_tracked_id)))
                                tolal_face_detected = sum(face_buffer[str(id)].values())
                                correct_person_name = max(zip(face_buffer[str(id)].values(), face_buffer[str(id)].keys()))[1]
                                print(f"tolal_face_detected: {tolal_face_detected}")
                                print(f"face_buffer: {face_buffer}")
                                temp_index = None
                                temp_index = [index for index in range(len(person_attendence)) \
                                                  if person_attendence[index]["EmpoloyeeName"]==correct_person_name]
                                temp_id = [index for index in range(len(person_attendence))\
                                                          if id in person_attendence[index]["EntranceId"]]
                                if len(temp_id)>0:
                                    if temp_id[0]!=temp_index[0]:
                                        max_id = max(person_attendence[temp_id[0]]["EntranceId"])
                                        min_id = min(person_attendence[temp_id[0]]["EntranceId"])
                                        #if id == max_id and max_id == min_id:
                                        person_attendence[temp_id[0]]["Availibilty"] = "Absent"
                                        person_attendence[temp_id[0]]["AttendenceTime"] = None
                                        id_to_be_remove = person_attendence[temp_id[0]]["EntranceId"]
                                        # Remove record from DB
                                        from datetime import datetime
                                        from time import localtime, strftime
                                        try:
                                            mm, dd, yy  = map(int, strftime("%m-%d-%Y", localtime()).split("-"))
                                            myquery = {"$and": [{'Date':{"$gte":\
                                                       datetime(yy, mm, dd,hour=0,minute=0,second=0)}},{"EntranceId": id_to_be_remove}]}
                                            result = daily.delete_one(myquery)
                                        except:
                                            print("Not able to delete record!!")
                                        person_attendence[temp_id[0]]["EntranceId"].remove(id_to_be_remove)
                                        
                                #else:
                                temp_index = temp_index[0]
                                size = 75 
                                #if tolal_face_detected >= buffer_length and person_attendence[temp_index]["Availibilty"] == "Absent":
                                if tolal_face_detected >= buffer_length:
                                    #person_attendence[correct_person_name] = "Present"
                                    if person_attendence[temp_index]["Availibilty"] == "Absent":
                                        from datetime import datetime
                                        curr_db_date = datetime.now()
                                        from time import localtime, strftime
                                        date_time = strftime("%m-%d-%Y %H:%M:%S", localtime())
                                        curr_date, curr_time = date_time.split(" ")
                                        person_attendence[temp_index]["Availibilty"] = "Present"
                                        person_attendence[temp_index]["AttendenceTime"] = curr_time
                                        # Record to be add
                                        person_crop_60x60 = cv2.resize(face_gamma,(size,size))
                                        IMAGE_STRING = base64.b64encode(cv2.imencode('.bmp', person_crop_60x60)[1]).decode("utf-8")
                                        #import zmq
                                        #import json 
                                        #context = None
                                        #socket = None
                                        #context = zmq.Context()
                                        #print("Connecting to hello world server !!")
                                        #socket = context.socket(zmq.REQ)
                                        #socket.connect("tcp://localhost:5556")
                                        #curr_timestamp = str(int(time.time()))
                                        #json_data = {"timestamp": curr_timestamp, "EmpoloyeeName": person_attendence[temp_index]["EmpoloyeeName"]}
                                        #json_str = json.dumps(json_data)
                                        #socket.send_string(json_str)
                                        #message = socket.recv()
                                        #print(f"message: {message}")
                                        #response_json = json.loads(message)
                                        #evidence_snippet_path = response_json["Evidence_snippets"]
                                        record = {\
                                                  "Date": curr_db_date,\
                                                  "EmpoloyeeName": person_attendence[temp_index]["EmpoloyeeName"],\
                                                  "Availibilty":  person_attendence[temp_index]["Availibilty"],\
                                                  "AttendenceTime": person_attendence[temp_index]["AttendenceTime"],\
                                                  "EntranceId": id,\
                                                  "AttendanceSnippet": IMAGE_STRING,\
                                                  "EvidenceClip": ""
                                                  }
                                        daily.insert_one(record)
                                        if id not in person_attendence[temp_index]["EntranceId"]:
                                            person_attendence[temp_index]["EntranceId"].append(id)
                                        if ((str(id) not in imgstring_buffer) and tolal_face_detected >=buffer_length):
                                            person_crop_60x60 = cv2.resize(face_gamma,(size,size))
                                            imgstring_buffer[str(id)] =  person_crop_60x60
                                seen_ids = list(imgstring_buffer.keys())
                                print(f"unique_ids: {unique_ids}")
                                print(f"seen_ids: {seen_ids}")
                                for temp_id in seen_ids:
                                    if int(temp_id) not in unique_ids:
                                        del imgstring_buffer[temp_id]
                                # Add snap on debug images
                                seen_ids = list(imgstring_buffer.keys())
                                print(f"unique_ids: {unique_ids}")
                                print(f"seen_ids: {seen_ids}")
                                count_temp = 0
                                if seen_ids:
                                    for temp_id in last_tracked_id:
                                        print(f"temp_id: {temp_id} and seen_ids: {seen_ids}")
                                        if str(temp_id) in seen_ids:
                                            tolal_face_detected = sum(face_buffer[str(temp_id)].values())
                                            temp_index = None
                                            if tolal_face_detected>buffer_length:
                                                temp_index = [index for index in range(len(person_attendence))\
                                                          if temp_id in person_attendence[index]["EntranceId"]]
                                                if len(temp_index) == 0:
                                                    temp_index = [index for index in range(len(person_attendence)) \
                                                                  if person_attendence[index]["EmpoloyeeName"]==correct_person_name][0]
                                                if len(temp_index)>0:
                                                    temp_index = temp_index[0]
                                                    temp_id_cv_img = imgstring_buffer[str(temp_id)]
                                                    p1 = (annotator.im.shape[1]-(2*size), (((count_temp*2)+1)*size))
                                                    p2 = (annotator.im.shape[1]-size, (((count_temp*2)+1)*size)+size)
                                                    annotator.im[p1[1]:p2[1], p1[0]:p2[0]] = temp_id_cv_img
                                                    # Adding Name Label
                                                    cv2.rectangle(annotator.im, p1, p2, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
                                                    correct_person_name = person_attendence[temp_index]["EmpoloyeeName"]
                                                    #correct_person_name = max(zip(face_buffer[str(temp_id)].values(), \
                                                    #                              face_buffer[str(temp_id)].keys()))[1]
                                                    lw = 2
                                                    tf = max(lw - 1, 1)
                                                    w, h = cv2.getTextSize(correct_person_name, 0, fontScale=lw/3, thickness=tf)[0]
                                                    outside = p1[1] - h >= 0
                                                    p2_ = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                                                    cv2.rectangle(annotator.im, p1, p2_, (0, 0, 255), -1, cv2.LINE_AA)  # filled
                                                    cv2.putText(annotator.im, correct_person_name, (p1[0], p1[1] - 2 \
                                                                if outside else p1[1] + h + 2), 0, lw / 3, (255, 255, 255), \
                                                                thickness=tf, lineType=cv2.LINE_AA)
                                                    #Adding Attendence time label
                                                    temp_index = [index for index in range(len(person_attendence))\
                                                                  if temp_id in person_attendence[index]["EntranceId"]][0]
                                                    attendence_time = person_attendence[temp_index]["AttendenceTime"]
                                                    w, h = cv2.getTextSize(attendence_time, 0, fontScale=lw/3, thickness=tf)[0]
                                                    outside = p1[1] + h >= 0
                                                    p1 = (p1[0], p2[1])
                                                    p2_ = p1[0] + w, p1[1] + h + 3
                                                    cv2.rectangle(annotator.im, p1, p2_, (0, 0, 255), -1, cv2.LINE_AA)  # filled
                                                    cv2.putText(annotator.im, attendence_time, (p1[0], p1[1] + h + 2), 0, lw / 3, (255, 255, 255), \
                                                                thickness=tf, lineType=cv2.LINE_AA)
                                                    count_temp += 1
                                            else:
                                                continue
                                print(f"person_attendence: {person_attendence}")
                                #print(f"attendence_record: {attendence_record}")
                                print(f"gender_label: {gender_label}")
                                #print(f"topk: {topk}")
                            if save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                       # elif iou_conf > 0.8 and int(cls)==1: 
                            if save_vid or save_crop or show_vid:  # Add bbox to image
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                label = gender_label
                                age_gender_label = f"{id} {label}: {conf:.2f}"
                                print(f"age_gender_label: {age_gender_label}")
                                if is_face_visible and names[c] == "head":
                                    #Id = label.strip().split(" ")[0]
                                    if label not in unique_seen_count_bilboard:
                                        unique_seen_count_bilboard[label] = []
                                    if id not in unique_seen_count_bilboard[label]:
                                        unique_seen_count_bilboard[label].append(id)
                                    #label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                    #    (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                    annotator.box_label(bboxes, age_gender_label, color=colors(c, True))
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')
            last_tracked_id = current_tracked_id
            print(f"last tracked Ids: {last_tracked_id}")
            im0 = annotator.result()
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 15, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
        #Send image to server to imshow
        im0 = cv2.resize(im0, (1920, 1080))
        #im0 = cv2.resize(im0, (1920, 1080),interpolation = cv2.INTER_NEAREST)
        sender.send_image(host_name, im0)
        #Export to csv
        import csv
        print(f"curr_time: {curr_time}")
        from time import localtime, strftime
        date_time = strftime("%Y-%m-%d %H:%M:%S", localtime())
        curr_date, curr_time = date_time.split(" ")
        if curr_time:
            saving_hh, saving_mm, _ = curr_time.split(":")
            print(f"saving_hh: {saving_hh}, saving_mm: {saving_mm}")
            if (f"{saving_hh}:{saving_mm}"=="09:45") and not is_csv_dumped:
                print("\n\n\ saving csv report!! \n\n!! ")
                with open(f'{curr_date}.csv', 'w') as f:
                    # using csv.writer method from CSV package
                    write = csv.DictWriter(f, fieldnames = field_names)
                    is_csv_dumped = True
                    write.writeheader()
                    write.writerows(person_attendence)
            elif (f"{saving_hh}:{saving_mm}"=="09:45") and is_csv_dumped:
                print("GOING to stop pipeline!!")
                sys.exit(0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
