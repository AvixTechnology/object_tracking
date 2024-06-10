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

Services and Topics:
- /object_detection_cmd: Action server for receiving object detection and following commands.
- /tracking_update: Published tracking updates, including target deviation and detected objects' information, for further processing or control actions.


Current Updates:
- 1.0.1: Update Kalman Filter for better tracking stability (Target GPS position and altitude)

Author: Hsinhua Lu
Date: 2024/4/7
Version: 0.6.1
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from avix_msg.msg import  ObjectDetection, ObjectDetections, MQ3State
import torch
import cv2
#from object_detection_util import KalmanFilter ,ReIDTrack
from object_detection_avix.object_detection_util import ReIDTrack
from cv_bridge import CvBridge, CvBridgeError
import numpy as np  
import time 




# for reading engine
from avix_utils.avix_utils.avix_enums import ObjectDetectionMode
from avix_utils.avix_utils import avix_common
from avix_utils.srv import ObjectDetectionStatus, EnableFunction, GetGimbalInfo

torch.cuda.device(0)
torch.manual_seed(23) #for cuda initialization bugs
torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

# status system 
from dataclasses import dataclass
@dataclass
class NodeState:
    enabled: bool = True
    is_intializing: bool = True
    is_detecting: bool = False
    detection_mode: ObjectDetectionMode = ObjectDetectionMode.Yolov8_BotSort
    
class TrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracking_node')
        # Subscribe to image coming
        self.image_subscriber = self.create_subscription(
            Image, 
            avix_common.KTG_EO_IMG, 
            self.image_callback, 
            10
        )

        self.master_subscriber = self.create_subscription(
            MQ3State, 
            avix_common.MQ3_STATUS, 
            self.master_status_callback, 
            10
        )
    
        # Publisher for the Detections
        self.box_publisher = self.create_publisher(
            ObjectDetections, 
            avix_common.OBJECT_DETECTIONS, 
            10
        ) 

        # state control
        self.state = NodeState()

        # Timer to reset tracking status
        self.tracking_timer = self.create_timer(3.0, self.reset_tracking_status)
        self.last_tracking_time = self.get_clock().now()

        #status service
        # Service for status checking
        self.status_service = self.create_service(
            ObjectDetectionStatus,
            avix_common.GIMBAL_TRACKING_STATUS,
            self.handle_get_status
        )

        # service to enable object detection
        self.enbale_service = self.create_service(
            EnableFunction,
            avix_common.ENABLE_OBJECT_DETECTION,
            self.handle_enable_object_detection
        )

        #check if enviornment is ok
        if(not torch.cuda.is_available()):
            self.get_logger().error(f'Pytorch has no cuda support, please reinstall')

        self.get_logger().info(f'Initializing Model...')

        # Initialize CV bridge
        self.bridge = CvBridge()

        self.model=ReIDTrack()
        # Generate a random image
        random_image = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

        # Convert the image to bgr8 format
        random_image_bgr8 = cv2.cvtColor(random_image, cv2.COLOR_RGB2BGR)

        self.model.track(random_image_bgr8)

        # initialization done
        self.state.is_intializing = False
         # Timer to reset following status
        self.detection_timer = self.create_timer(3.0, self.reset_following_status)
        self.last_tracking_time = self.get_clock().now()

        self.detection = ObjectDetection()
        self.objects_data = ObjectDetections()

        # intialize the client
        self.cli_gimbal_info = self.create_client(GetGimbalInfo, avix_common.KTG_CAMERA_INFO)
        
        # Wait for services to start
        self.wait_for_services()

        # request the camera info
        self.get_camera_info()

        self.get_logger().info(f'*******Object Dtection Node started (V1.0.1)**********')
    
    # ============service related============
    # region Service Related
    def wait_for_services(self):
        while not self.cli_gimbal_info.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('[Gimbal info] Service not available, waiting again... Please boot it up if have not!')

    def get_camera_info(self):
        request = GetGimbalInfo.Request()  
        future = self.cli_gimbal_info.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        self.input_width = response.resolution_x
        self.input_height = response.resolution_y
        

    def reset_following_status(self):
        if not self.state.is_detecting:
            return # no need to reset
        current_time = self.get_clock().now()
        if (current_time - self.last_tracking_time).nanoseconds > 3 * 1e9:
            self.state.is_detecting = False

    def handle_get_status(self, request, response):
        response.enabled = self.state.enabled
        response.is_detecting = self.state.is_detecting
        response.detection_mode = self.state.detection_mode.value
        response.is_intializing = self.state.is_intializing
        return response
    
    def handle_enable_object_detection(self, request, response):
        if self.state.enabled == request.enable:
            if request.enabled:
                response.error_code = 100
            else:
                response.error_code = 101
            response.success = False
            return response
        
        if request.enable:
            if self.state.is_intializing:
                response.success = False
                response.error_code = 1
                response.message = 'Tracking system is not initialized.'
                self.get_logger().warn("Tracking system is not initialized. Skipping the enable service request.")
                return response
            
            self.state.enabled = True
        else:
            self.state.enabled = False
        
        response.error_code = 0
        response.success = True
        return response
    

    def master_status_callback(self, msg):  
        # check if the targetid is the same
        if self.state.enabled != msg.detection_enabled:
            self.get_logger().warn(f'[SEVERE] Following status does not match, current {self.state.following_enabled} vs master {msg.following_enabled}. Changing to it...')
            self.state.following_enabled = msg.following_enabled
    # endregion

    # ============image related============
    # region image related
    def image_callback(self, msg):
        start =time.time()
        # not run the model if not initialized
        if self.state.is_initializing:
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
        
        results , _ , _ = self.model.track(cv_image)
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
            c,  id = int(tcls), int(tid) %128
            x1,y1,x2,y2=tlbr[0],tlbr[1],tlbr[2],tlbr[3]


            self.detection.id = id  # Assign the detection ID
            self.detection.bbox = tlbr  # Replace with actual bbox coordinates
            self.detection.class_type = c  # Replace with actual class type
            self.detection.confidence = float(0)  # Replace with actual confidence
            num_detections +=1

            self.objects_data.detections.append(self.detection)

        #self.get_logger().info(f'object data: {objects_data}')
        if(num_detections>0): 
            self.objects_data.num_detections = num_detections
            self.box_publisher.publish(self.objects_data)

        self.last_tracking_time = self.get_clock().now()
        self.state.is_detecting = True

        end =time.time()
        self.get_logger().info(f"all time : { end - start} ")
    # endregion




def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
