
from ultralytics import YOLO
from copy import deepcopy
import torch
torch.cuda.set_device(0) # Set to your desired GPU number
from ultralytics.utils.plotting import Annotator, colors

import cv2
import numpy as np
import pandas as pd
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
        self.iou_thres = 0.2
        self.device = 0
        self.view_img = False
        self.classes = [0]
        self.agnostic_nms = False
        self.augment = False
        self.fp16 = True
        self.fuse = False
        self.project = 'runs/track'
        self.name = 'MOT17-01'
        self.trace = False
        self.hide_labels_name = False
        self.default_parameters = False
        self.save_frames = False
        self.track_high_thresh = 0.7
        self.track_low_thresh = 0.4   
        self.new_track_thresh = 0.3
        self.track_buffer = 30
        self.match_thresh = 0.9
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.cmc_method = 'sparseOptFlow'
        self.ablation = False
        self.with_reid = False
        self.fast_reid_config = r"/home/nvidia/Documents/avix/inference_dependency/BoT-SORT/fast_reid/configs/MSMT17/sbs_S50.yml"
        self.fast_reid_weights = r"/home/nvidia/Documents/avix/inference_dependency/BoT-SORT/pretrained/msmt_sbs_S50.pth"
        self.proximity_thresh = 0.05
        self.appearance_thresh = 0.15

class ReIDTrack():
    def __init__(self) -> None:
        opt = botsortConfig()
        package_dir = get_package_share_directory('object_detection_avix')
        
        # Construct the full path to the .engine file
        #engine_path = os.path.join(package_dir, 'yolov8s_fp16_736x1280.engine')
        engine_path = os.path.join(package_dir, 'opt.engine')
        self.model = YOLO(engine_path,task="detect")
        self.tracker = BoTSORT(opt, frame_rate=10.0)

        #save csv file 
        self.yolo_data = []
        self.BotSort_data = []
       
        #recording file
        # self.yolo_paint_file = open('yolo_paint_results.txt', 'w')
        # self.yolo_all_file = open('yolo_all_results.txt', 'w')
        # self.yolo_model_file = open('yolo_model_results.txt', 'w')

        

    def track(self,frame):
        self.yolo_data = []
        self.BotSort_data = []
        tic = time.time()
        results = self.model.predict(source = frame ,conf=0.3, classes=[0,1,2,3],imgsz=(736,1280),verbose=False, half = True, device="cuda:0",)
        #print(results[0].boxes)    
        toc = time.time()
        predict_time=toc - tic
        # print(f"predict time {toc - tic}")    
        boxes = results[0].boxes
        

        bboxes = boxes.xyxy.cpu().numpy()  # Convert tensors to numpy arrays
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        #save yolo data  with bboxes, scores, classes as a csv file 
        
        self.yolo_data.append( [predict_time/1.0 , bboxes, scores,  classes])
       


        # Constructing the 2D list
        detection_list = np.column_stack((bboxes, scores, classes, np.zeros(len(bboxes))))
        online_targets = self.tracker.update(detection_list,results[0].orig_img)
        toc2 = time.time()

        update_time=toc2-toc
        # print(f"update time {toc2 - toc}")  
        
        annotator = Annotator(
        deepcopy(results[0].orig_img),
            line_width,
            font_size,
            font,
            pil,  # Classify tasks default to pil=True
            example=results[0].names
        )
        #save the BotSort data as a csv file
        
        for t in online_targets:
            tlbr = t.tlbr
            tid = t.track_id %128
            tcls = t.cls
            c,  id = int(tcls), int(tid)
            label =  ('' if id is None else f'id:{id} ') + results[0].names[c]
            annotator.box_label(tlbr, label, color=colors(c, True))  
            self.BotSort_data.append([update_time, results[0].names[c], id,  tlbr])

        annotated_frame = annotator.result()
        annotated_frame = cv2.resize(annotated_frame,(640,480))
        
        #cv2.imshow("test1",annotated_frame)
        # toc3=time.time()
        # painttime=toc3-toc2
        # print("paint time: ", toc3-toc2)
        #cv2.waitKey(1)
        
        # self.Botsort_file.write(str(updatetime)+ ", ")
        # self.yolo_paint_file.write(str(painttime)+ ", ")
        # self.yolo_model_file.write(str(predict_time)+ ", ")

        #save BotSort data and yolodata
        # self.yolo_data_df = pd.DataFrame( self.yolo_data ,columns=['time', 'bboxes', 'scores', 'classes'])
        # self.BotSort_data_df = pd.DataFrame( self.BotSort_data , columns=['time', 'cls', 'id', 'tlbr'])
        # self.BotSort_data_df.to_csv('bot_sort_results.csv', index=False)
        # self.yolo_data_df.to_csv('yolo_results.csv', index=False)
        # toc3=time.time()
        # print("all time :", toc3 - tic )
        
        
        return online_targets ,self.yolo_data ,self.BotSort_data , annotated_frame


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
