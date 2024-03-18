
from ultralytics import YOLO
from copy import deepcopy
import torch
torch.cuda.set_device(0) # Set to your desired GPU number
from ultralytics.utils.plotting import Annotator, colors

import cv2
import numpy as np
import time,sys,os
current_dir = os.path.dirname(os.path.abspath(__file__))
bot_sort_path = os.path.join(current_dir, 'BoT-SORT')
sys.path.append(bot_sort_path)
from tracker.mc_bot_sort import BoTSORT
#basic setting
line_width=None
font_size=None
font='Arial.ttf'
pil = False
rtsp_url = "mot4.mp4"


# for reading engin
from ament_index_python.packages import get_package_share_directory

class botsortConfig():
    def __init__(self):
        self.source = 'inference/images'
        self.weights = ['yolov7.pt']
        self.benchmark = 'MOT17'
        self.split_to_eval = 'test'
        self.img_size = 1280
        self.conf_thres = 0.2
        self.iou_thres = 0.7
        self.device = 0
        self.view_img = False
        self.classes = [0]
        self.agnostic_nms = False
        self.augment = False
        self.fp16 = False
        self.fuse = False
        self.project = 'runs/track'
        self.name = 'MOT17-01'
        self.trace = False
        self.hide_labels_name = False
        self.default_parameters = False
        self.save_frames = False
        self.track_high_thresh = 0.5
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.6
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.cmc_method = 'sparseOptFlow'
        self.ablation = False
        self.with_reid = True
        self.fast_reid_config = r"/home/avix/tracking_modules/BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml"
        self.fast_reid_weights = r"/home/avix/tracking_modules/BoT-SORT/pretrained/mot17_sbs_S50.pth"
        self.proximity_thresh = 0.4
        self.appearance_thresh = 0.15

class ReIDTrack():
    def __init__(self) -> None:
        opt = botsortConfig()
        package_dir = get_package_share_directory('object_detection_avix')
        
        # Construct the full path to the .engine file
        engine_path = os.path.join(package_dir, 'yolov8s736x1280.engine')
        self.model = YOLO(engine_path,task="detect")
        self.tracker = BoTSORT(opt, frame_rate=30.0)

    def track(self,frame):
        tic = time.time()
        results = self.model.predict(source = frame ,conf=0.6,classes=[0,1,2,3],imgsz=(736,1280),verbose=False)
        #print(results[0].names)    
        toc = time.time()
        #print(f"predict time {toc - tic}")    
        boxes = results[0].boxes

        bboxes = boxes.xyxy.cpu().numpy()  # Convert tensors to numpy arrays
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        # Constructing the 2D list
        detection_list = np.column_stack((bboxes, scores, classes, np.zeros(len(bboxes))))
        online_targets = self.tracker.update(detection_list,results[0].orig_img)
        toc2 = time.time()
        #print(f"update time {toc2 - toc}")  
        annotator = Annotator(
            deepcopy(results[0].orig_img),
            line_width,
            font_size,
            font,
            pil,  # Classify tasks default to pil=True
            example=results[0].names
        )
        for t in online_targets:
            tlbr = t.tlbr
            tid = t.track_id
            tcls = t.cls
            c,  id = int(tcls), int(tid)
            label =  ('' if id is None else f'id:{id} ') + results[0].names[c]
            annotator.box_label(tlbr, label, color=colors(c, True))  
        

        annotated_frame = annotator.result()
        annotated_frame = cv2.resize(annotated_frame,(640,384))
        
        cv2.imshow("test1",annotated_frame)
        cv2.waitKey(1)
        
        return online_targets


class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
