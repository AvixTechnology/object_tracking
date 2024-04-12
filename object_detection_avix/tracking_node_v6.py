"""
TrackingNode is designed for object tracking and localization within a robotic system, 
leveraging data fusion from multiple sensors. It subscribes to image streams, gimbal state, GPS data, and tracking commands, 
processing this information to maintain a lock on designated targets identified by an object detection model. The node utilizes 
the YOLO object detection algorithm for identifying targets within visual data, enhanced with Kalman Filtering and ReID (Re-Identification) 
tracking for improved precision and stability in tracking movements. Additionally, it computes target localization by translating 
visual tracking data into GPS coordinates, using the platform's gimbal orientation and GPS state to estimate the target's position.

The node is designed to interface with various components of a UAV or robotic platform, including:
- Image data from a gimbal-mounted camera for object detection and tracking.
- Gimbal orientation data for accurate target angle estimation.
- GPS data from the platform for calculating the target's geolocation.
- Custom action servers for receiving tracking and following commands, allowing for dynamic control over the tracking process.

Features include:
- Subscription to ROS topics for real-time image data, gimbal info, and platform state.
- Action server implementation for processing object detection and following commands.
- Integration of the YOLO model for object detection, supported by Kalman Filtering and ReID tracking for robust target tracking.
- Calculation of target GPS coordinates from visual data, utilizing the platform's orientation and position.
- Publication of tracking updates, including target deviation and detected objects' information, for further processing or control actions.

This node is a crucial component for applications requiring autonomous tracking and localization capabilities, such as search and rescue, surveillance, 
and target following, providing a versatile and reliable solution for integrating complex sensor data into actionable insights.

Dependencies:
- ROS 2 (Robot Operating System)
- OpenCV for image processing
- PyTorch and ultralytics YOLO for object detection
- CvBridge for converting between ROS image messages and OpenCV images
- NumPy for numerical computations
- Avix related package: avix_msg, avix_action

Current Updates:
- 0.6.1: Following required system (GPS)

Author: Hsinhua Lu
Date: 2024/4/7
Version: 0.6.1
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from avix_action.action import ObjectDetectionCMD
from std_msgs.msg import Bool, Int32MultiArray, Float32MultiArray, Int32, ByteMultiArray
from ultralytics import YOLO
from avix_msg.msg import TrackingUpdate, MavlinkState, ObjectDetection, Yolodata, BotSortdata, ObjectDetections, TargetGPS, TrackingCommand, InfInfo, GimbalInfo
import torch
import cv2
import struct
#from object_detection_util import KalmanFilter ,ReIDTrack
from object_detection_avix.object_detection_util import KalmanFilter ,ReIDTrack
from cv_bridge import CvBridge, CvBridgeError
import numpy as np  
from rclpy.action import ActionServer
import math
import time 



# for reading engin
import os
from ament_index_python.packages import get_package_share_directory

torch.cuda.device(0)
torch.manual_seed(23) #for cuda initialization bugs
torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
# Prepare the packet header
packet_header = bytes([0x41, 0x76, 0x69, 0x78])
# Sender ID: 0x04 (AI)
sender_id = bytes([0x04])

class TrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracking_node')
        # Subscribe to necessary topics
        self.target_subscriber = self.create_subscription(Int32, '/object_detection/tracking_target_id', self.target_id_callback, 10)
        self.image_subscriber = self.create_subscription(Image, '/ktg_gimbal/image_raw', self.image_callback, 10)
        #self.enable_subscriber = self.create_subscription(TrackingCommand, '/icp_interface/tracking_cmd', self.status_callback, 10)
        self.cmd_action_server =  ActionServer(
            self,
            ObjectDetectionCMD,
            'object_detection/cmd',
            self.command_callback)
        
        # inf interface
        # self.inf_subscriber = self.create_subscription(Inf_info, '/inf_interface/gps_info', self.gps_inf_callback, 10)
            
        # ktg interface
        self.gimbal_subscriber = self.create_subscription(GimbalInfo, '/gimbal/info', self.gimbal_callback, 10)

        # avix_mavros interface
        self.mav_subscriber = self.create_subscription(MavlinkState, 'avix_mavros/state', self.gps_mavlink_callback, 10)

        # Publisher for the deviation 
        self.deviation_publisher = self.create_publisher(TrackingUpdate, '/object_detection/target_deviation', 1)
        self.box_publisher = self.create_publisher(ObjectDetections, '/object_detection/detections', 10) # bytes array is not working yet
        self.targetGPS_publisher = self.create_publisher(TargetGPS, '/object_detection/target_gps', 10)
        self.initializing = True

        # initialize the parameter
        # self.declare_parameter('tracking_enabled', False)
        #self.tracking_enabled = self.get_parameter('tracking_enabled').value

        self.declare_parameter('confidence', 0.3)
        self.confidence = self.get_parameter('confidence').value

        self.declare_parameter('input_width', 1280)
        self.input_width = self.get_parameter('input_width').value

        self.declare_parameter('input_height', 720)
        self.input_height = self.get_parameter('input_height').value

        self.declare_parameter('target_object', [0])
        self.target_objects = self.get_parameter('target_object').value
        
        #check if enviornment is ok
        if(not torch.cuda.is_available()):
            self.get_logger().warn(f'Pytorch has no cuda support, please reinstall')

        # Get the directory where your package is installed
        #package_dir = get_package_share_directory('object_detection_avix')
        
        # Construct the full path to the .engine file
        #engine_path = os.path.join(package_dir, 'yolov8n.engine')

        self.get_logger().info(f'Initializing Model...')

        #start up the tensorrt 

        
        # Initialize CV bridge
        self.bridge = CvBridge()

        self.result=ReIDTrack()
        # Generate a random image
        random_image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

        # Convert the image to bgr8 format
        random_image_bgr8 = cv2.cvtColor(random_image, cv2.COLOR_RGB2BGR)

        self.result.track(random_image_bgr8)

        # initialize the tracking variable
        self.target_id = -1 # for following
        self.KF_initailized = False

        self.initializing = False

        # state of this system
        self.state_tracking = False
        self.state_following = False
        # Initialize your YOLO model
        self.get_logger().info(f'Model Initialized...')

        # flag for mq3 (0) or inf(1)
        self.fc_type = 0

        # node created
        self.detection = ObjectDetection()
        self.objects_data = ObjectDetections()

        # for recording
        # self.track_file = open('track_results.txt', 'w')
        self.frame_count = 0

        
        self.get_logger().info(f'Object Detection Node created')


    def image_callback(self, msg):
        start =time.time()
        # not run the model if not initialized
        if self.initializing:
            return

        # not run the model if tracking is disabled
        if not self.state_tracking:
            return
        
        # Convert ROS Image message to CV2 format and pass it to model
        try:
            # Convert the ROS Image message to a CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting ROS Image to OpenCV: {e}')
            return
        # readtime=time.time()
        # print("readtime: ", readtime-start)
        # Check the size of the image
        height, width = cv_image.shape[:2]
        if width != self.input_width or height != self.input_height:
            # resize the image
            self.get_logger().warn(
               f"Received image of dimensions ({width}, {height}), which does not match expected dimensions ({self.input_width}, {self.input_height}). Resizing image.")
            cv_image = cv2.resize(cv_image, (self.input_width, self.input_height))

        # do the tracking
        
        results , yolodata, Botsortdata = self.result.track(cv_image)
        # Prepare the objects' data

        num_detections = 0
        # analyze the results
        # if it is following object

        # track_time =time.time()
        # print("tracktime: " , track_time-readtime)
        self.objects_data = ObjectDetections()

        for t in results:
            self.detection = ObjectDetection()
            tlbr = t.tlbr
            tid = t.track_id
            tcls = t.cls
            c,  id = int(tcls), int(tid)
            x1,y1,x2,y2=tlbr[0],tlbr[1],tlbr[2],tlbr[3]
            if(id  == self.target_id and self.state_following):
                # check if it is inside the center of the image
                x_center = (x1+x2)/2
                y_center = (y1+y2)/2
                self.follow(x_center,y_center,x2-x1,y2-y1)
                 # check if it is within 30% around center
                if(x_center<self.input_width/2*1.3 and x_center>self.input_width/2*0.7 and y_center<self.input_height/2*1.3 and y_center>self.input_height/2*0.7):
                    #send the gps data
                    deduced_lat, deduced_long, deduced_alt = self.find_location() # next we need to use the angle to deduce the right one
                    #print it out
                    self.get_logger().info(f'GPS: {deduced_long},{deduced_lat},{deduced_alt}')
                    gps_msg = TargetGPS()
                    gps_msg.target_longitude = deduced_long
                    gps_msg.target_latitude = deduced_lat
                    gps_msg.target_altitude = deduced_alt
                    gps_msg.estimate_source = self.ranging_flag

                    #TODO a kalman filter here could suit the requirement, but first need to study the fluctuation
                    self.targetGPS_publisher.publish(gps_msg)

            self.detection.id = id  # Assign the detection ID
            self.detection.bbox = tlbr  # Replace with actual bbox coordinates
            self.detection.class_type = c  # Replace with actual class type
            self.detection.confidence = float(0)  # Replace with actual confidence
            num_detections +=1

            self.objects_data.detections.append(self.detection)

        # save the yolo data and BotSort data
        # print (yolodata)
        # print(Botsortdata)
        # self.frame_count += 1
        # for i in range(len(yolodata[0][1])): 
        #     self.yolo_data= Yolodata()
        #     self.yolo_data.frameth=self.frame_count
        #     self.yolo_data.spendtime = yolodata[0][0]
        #     self.yolo_data.bbox = yolodata[0][1][i] # TODO check bbox type 
        #     self.yolo_data.score = float(yolodata[0][2][i])
        #     self.yolo_data.class_type = int(yolodata[0][3][i])
        #     self.objects_data.yolo_data.append(self.yolo_data)


        # for j in range(len(Botsortdata)):
        #     self.botsort_data = BotSortdata()
        #     self.botsort_data.frameth=self.frame_count
        #     self.botsort_data.spendtime = Botsortdata[j][0]
        #     self.botsort_data.class_type = Botsortdata[j][1]
        #     self.botsort_data.id = Botsortdata[j][2]
        #     self.botsort_data.bbox = Botsortdata[j][3]
        #     self.objects_data.botsort_data.append(self.botsort_data) # TODO check bbox type 


            


        #self.get_logger().info(f'object data: {objects_data}')
        if(num_detections>0): 
            self.box_publisher.publish(self.objects_data)


        end =time.time()
        #print("detection all  time:", end - start)
        # self.get_logger().info(f"all time : { end - start} ")
        #record the time
        # finish_time=time.time()
        # spend_time = finish_time - track_time
        # self.track_file.write(str(spend_time)+ ", ")
            

    def follow(self, cx, cy,size_x,size_y):
        if(not self.KF_initailized):
            self.initialize_KF(cx,cy)

        filtered_x = np.dot(self.H,  self.KF_x.predict())[0]
        filtered_y = np.dot(self.H,  self.KF_y.predict())[0]

        # publish the deviation
        deviation = TrackingUpdate()
        deviation.deviation_x = filtered_x-self.input_width/2
        deviation.deviation_y = filtered_y-self.input_height/2
        deviation.resolution_x = float(self.input_width)
        deviation.resolution_y = float(self.input_height)
        deviation.size_x = float(size_x)
        deviation.size_y = float(size_y)
        self.deviation_publisher.publish(deviation)

        self.KF_x.update(cx)
        self.KF_y.update(cy)

    def initialize_KF(self,x,y):
        dt = 1/15 # assume 15 fps
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        self.H = np.array([1, 0, 0]).reshape(1, 3)
        Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
        R = np.array([0.5]).reshape(1, 1)
        x_0 = np.array([x,0,0])
        y_0 = np.array([y,0,0])
        #del self.KF_x
        self.KF_x = KalmanFilter(F = F, H = self.H, Q = Q, R = R, x0=x_0)
        self.KF_y = KalmanFilter(F = F, H = self.H, Q = Q, R = R, x0=y_0)
        self.KF_initailized = True

    def command_callback(self, goal_handle):
        self.get_logger().info('Changing Status Based on command')
        result = ObjectDetectionCMD.Result()
        tracking_enabled = goal_handle.request.tracking_enabled
        following_enabled = goal_handle.request.following_enabled
        following_id = goal_handle.request.following_id
        #self.get_logger().info(f'{self.initializing}')
        self.get_logger().info(f'{tracking_enabled}')
        
        self.get_logger().info(f'follow enabled:{following_enabled}')
        self.get_logger().info(f'follow id :{following_id}')
        

        if self.initializing:
            self.get_logger().warn(f'Object Detection Model still initializing, please send command later')
            goal_handle.abort()
            result.error_code = 1
            return result
        
        if tracking_enabled:
            self.state_tracking = True

            if(following_enabled):
                self.state_following = True
                self.target_id = following_id
                self.get_logger().info(f'id change to {self.target_id} ')
            else:
                self.state_following = False
                self.target_id = -1
        self.get_logger().info('Success!!!!...')
        goal_handle.succeed()

        result.tracking_enabled = tracking_enabled
        result.following_enabled = following_enabled
        result.following_id = following_id
        result.error_code = 0

        return result
    
    def gimbal_callback(self, msg):
        self.gimbal_info = msg

        # update the require information
        self.gimbal_pitch = msg.pitch_angle
        self.gimbal_yaw = msg.yaw_angle
        self.ranging_flag = msg.ranging_flag
        self.target_distance =msg.target_distance

    def gps_mavlink_callback(self, msg):
        self.gps_mavlink_info = msg

        # update the require information
        self.longitude = msg.longitude
        self.latitude = msg.latitude
        self.altitude = msg.altitude
        self.rel_alt = msg.relative_altitude
        self.heading = msg.heading
        self.uav_pitch = msg.pitch
        self.uav_roll = msg.roll

    def gps_inf_callback(self,msg):
        self.gps_inf_info = msg

        # update the require information
        self.longitude = msg.longitude
        self.latitude = msg.latitude
        self.altitude = msg.altitude

        self.uav_roll = msg.roll
        self.uav_pitch = msg.pitch
        self.headng = msg.yaw

    def find_location(self):
        # Constants
        EARTH_RADIUS = 6371000  # in meters
        try:
            # Step 1: Calculate absolute gimbal orientation
            abs_gimbal_pitch = self.gimbal_pitch
            abs_gimbal_yaw = self.heading + self.gimbal_yaw
            self.ranging_flag=False
            # With ranging module
            if(self.ranging_flag):
                # Step 2: Calculate the distance to the object
                pitch_radians = math.radians(-abs_gimbal_pitch)
                distance_to_object =  self.target_distance 

                self.get_logger().info(f'distance:  {distance_to_object} ')

                # Step 3: Calculate the GPS location of the object
                yaw_radians = math.radians(abs_gimbal_yaw)
                delta_north = distance_to_object * math.cos(yaw_radians)
                delta_east = distance_to_object * math.sin(yaw_radians)

                delta_latitude = (delta_north / EARTH_RADIUS) * (180 / math.pi)
                delta_longitude = (delta_east / EARTH_RADIUS) * (180 / math.pi) / math.cos(math.radians(self.latitude))

                object_latitude = self.latitude + delta_latitude
                object_longitude = self.longitude + delta_longitude
                
                # Calculate the object's altitude
                object_altitude = self.altitude - (distance_to_object * math.sin(pitch_radians))

            # Without ranging module
            else:
                if(self.fc_type == 0):
                    rel_alt = self.altitude
                elif(self.fc_type == 1):
                    rel_alt = self.rel_alt
                
                # Assuming object is at ground level
                object_altitude = 0  # Object is at ground level
                rel_alt = 1
                # We still need to calculate the object's GPS location as before
                # Since the object is assumed at ground level, we use rel_alt for calculations
                # Calculate distance to object assuming the pitch angle points to the horizon
                
                distance_to_object = rel_alt / math.tan(math.radians(-abs_gimbal_pitch))
                self.get_logger().info(f'distance : {distance_to_object}')
                yaw_radians = math.radians(abs_gimbal_yaw)
                delta_north = distance_to_object * math.cos(yaw_radians)
                delta_east = distance_to_object * math.sin(yaw_radians)

                delta_latitude = (delta_north / EARTH_RADIUS) * (180 / math.pi)
                delta_longitude = (delta_east / EARTH_RADIUS) * (180 / math.pi) / math.cos(math.radians(self.latitude))

                object_latitude = self.latitude + delta_latitude
                object_longitude = self.longitude + delta_longitude

        except AttributeError as e:
            self.get_logger().warn(f'some property has not been intialized: {e}')
            return 0.0,0.0,0.0

        return object_latitude, object_longitude, object_altitude/1.0

        

    def target_id_callback(self, msg):
        # Assuming msg.data contains the ID of the target to track
        self.target_id = msg.data
        self.get_logger().info(f'Received target ID: {self.target_id}')

    def publish_deviation(self, deviation_x, deviation_y):
        msg = Float32MultiArray()
        msg.data = [deviation_x, deviation_y]
        self.deviation_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
