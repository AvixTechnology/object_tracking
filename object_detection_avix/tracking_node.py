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
from avix_utils.msg import  ObjectDetection, ObjectDetections, MQ3State, GimbalControl
from std_msgs.msg import Int32
import torch
import cv2
#from object_detection_util import KalmanFilter ,ReIDTrack
from object_detection_avix.object_detection_util import ReIDTrack
from cv_bridge import CvBridge, CvBridgeError
import numpy as np  
import time 
import math




# for reading engine
from avix_utils_py.avix_enums import ObjectDetectionMode
from avix_utils_py import avix_common
from avix_utils.srv import ObjectDetectionStatus, EnableFunction, GetGimbalInfo

torch.cuda.device(0)
torch.manual_seed(23) #for cuda initialization bugs
torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

# status system 
from dataclasses import dataclass
@dataclass
class NodeState:
    enabled: bool = True
    is_intializing: bool = True
    is_detecting: bool = False
    detection_mode: ObjectDetectionMode = ObjectDetectionMode.Yolov8_BotSort

LOSE_TRACKING_FRAME_THRESHOLD = 15
LOSE_TRACKING_FRAME_THRESHOLD_AFTER_ZOOM = 10
LOSE_TRACKING_DISTANCE_THRESHOLD = 0.3 # 30% of the image width
LOSE_TRACKING_SIZE_THRESHOLD = 0.6 # 80% of the size changed


class TrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracking_node')
        print("flag1")

        # declare parameters
        self.declare_parameter('enable_spatial_REID', True) # won't move if target is too deviated
        self.spatial_REID_enabled = self.get_parameter('enable_spatial_REID').value
        
        # Subscribe to image coming

        # Create a QoS profile with a queue size of 1 (keep only the latest message)
        qos_profile = QoSProfile(
            depth=1,  # Queue size
            history=QoSHistoryPolicy.KEEP_LAST,  # Keep only the last message
            durability=QoSDurabilityPolicy.VOLATILE,  # Volatile to avoid storing messages if no subscriber
            reliability=QoSReliabilityPolicy.BEST_EFFORT  # Ensure reliable delivery
        )
        self.image_subscriber = self.create_subscription(
            Image, 
            avix_common.KTG_EO_IMG, 
            self.image_callback,   
            qos_profile
        )
        self.eo_zoom=1
        self.ktg_info_subscriber = self.create_subscription(
            Image, 
            avix_common.KTG_INFO, 
            self.ktg_info_callback,   
            10
        )

        self.master_subscriber = self.create_subscription(
            MQ3State, 
            avix_common.MQ3_STATUS, 
            self.master_status_callback, 
            10
        )

        if self.spatial_REID_enabled:
            self.currentID_subscriber = self.create_subscription(
                Int32, 
                avix_common.MQ3_CURRENT_TARGET_ID, 
                self.currentID_callback, 
                10
            )


    
        # Publisher for the Detections
        self.box_publisher = self.create_publisher(
            ObjectDetections, 
            avix_common.OBJECT_DETECTIONS, 
            10
        ) 
        if self.spatial_REID_enabled:
            # Publisher for zoom out 
            self.control_publisher = self.create_publisher(
                GimbalControl,
                avix_common.KTG_CONTROL,
                10
            )

        # Publisher for emergency change of ID
        self.ID_publisher = self.create_publisher(
            Int32, 
            avix_common.REID_BACKUP, 
            10
        ) 

        # state control
        self.state = NodeState()

        # Timer to reset tracking status
        self.tracking_timer = self.create_timer(3.0, self.reset_following_status)
        self.last_tracking_time = self.get_clock().now()

        #status service
        # Service for status checking
        self.status_service = self.create_service(
            ObjectDetectionStatus,
            avix_common.OBJECT_DETECTION_STATUS,
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

        self.get_logger().info(f'Initializing Model1112...')

        # Initialize CV bridge
        self.bridge = CvBridge()

        self.model=ReIDTrack(self.get_logger())
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

        # REID related
        self.currentID = -1
        self.lastframe_istracking = -1 
        self.imageCount = 0 # for skipping frames
        self.time_elapsed = 0 # for skipping frames
        self.lastframe_istracking_after_zoom_out=-1
        self.zooming_flag=False
      
        # -1 means is tracking nothing to be worryed
        # 0 means lose the target in first frame
        # 1,2,3 means lose the target after 1,2,3 frame

        # intialize the client
        self.cli_gimbal_info = self.create_client(GetGimbalInfo, avix_common.KTG_CAMERA_INFO)
        
        # Wait for services to start
        self.wait_for_services()
        self.init_gimbal_info =False
        # request the camera info
        self.get_camera_info()

        self.get_logger().info(f'*******Object Detection Node startedasd123 (V1.0.1)**********')

    
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
        self.fps = response.fps
        self.init_gimbal_info= True

        self.get_logger().info(f'Camera info get! resolution: ({self.input_width}.{self.input_height})')


        

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
        self.get_logger().info(f"Incoming request {request}")
        if self.state.enabled == request.enable:
            if request.enable:
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
        self.get_logger().info(f"response is {response}")
        return response
    
    def ktg_info_callback(self,msg):
        self.eo_zoom=msg.eo_zoom

    def master_status_callback(self, msg):  
        # check if the enable status
        if self.state.enabled != msg.detection_enabled:
            self.get_logger().warn(f'[SEVERE] Following status does not match, current {self.state.enabled} vs master {msg.detection_enabled}. Changing to it...')
            self.state.enabled = msg.detection_enabled

        # check the target id
        if self.currentID != msg.current_following_id:
            self.get_logger().warn(f'[SEVERE] Target ID does not match, current {self.currentID} vs master {msg.current_following_id}. Changing to it...')
            self.currentID = msg.current_following_id

    # endregion

    # ============REID related============
    # region REID related
    def currentID_callback(self, msg):
        self.currentID = msg.data
        self.get_logger().info(f"Current ID: {self.currentID}")

    def retrieve_target(self,results, zoom_flag=False):
        # speed up
        if results is None:
            return None
        
        # we go through the detection to see if there is one close enough to the id KF buffer
        kf_target = self.model.find_id_kf_loc(self.currentID)
        
        # if not, we just return None
        if kf_target is None:
            return None
        
        # for the zoom out action
        if self.eo_zoom==1 or not zoom_flag:
            zoom_change=1
        else:
            zoom_change=(self.eo_zoom)**(1/2)



        (x1,y1,x2,y2) = kf_target # we got the tlbr of the target
        target_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        target_size = (x2 - x1, y2 - y1)
        passed_object = {}
        for t in results:
            (tx1,ty1,tx2,ty2) = t.tlbr
            tid = t.track_id
            detection_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
            detection_size = (tx2 - tx1, ty2 - ty1)

            # Calculate the Euclidean distance between the centers
            distance = math.sqrt((detection_center[0] - target_center[0]) ** 2 +
                                 (detection_center[1] - target_center[1]) ** 2)
            
            # Calculate size changes
            width_change = abs(detection_size[0] - target_size[0]) / target_size[0] /zoom_change
            height_change = abs(detection_size[1] - target_size[1]) / target_size[1] /zoom_change

            # Check if the size change exceeds the threshold
            size_changed = width_change > LOSE_TRACKING_SIZE_THRESHOLD or height_change > LOSE_TRACKING_SIZE_THRESHOLD

            # Check if the distance exceeds the threshold
            moved= distance > LOSE_TRACKING_DISTANCE_THRESHOLD * self.input_width
            # print("distance " , distance)
            # print("LOSE_TRACKING_DISTANCE_THRESHOLD * self.input_width",LOSE_TRACKING_DISTANCE_THRESHOLD * self.input_width)
            # print("detection_center",detection_center)
            # print("target_center", target_center)

            # print("moved", moved)
            # print("detection_size",detection_size)
            # print("width_change" , width_change)
            # print("height_change", height_change)
            # print("size_changed", size_changed)
            if not size_changed and not moved:
                # The detection is within acceptable thresholds
                passed_object[tid] = distance
        

        print(passed_object)
        
        # check if passed_object is none
        if passed_object:
            min_id = min(passed_object,key=passed_object.get)
            return min_id

        return None

    def zoom_out(self):
        """
        Zoom out the camera if the target is lose tracking after LOSE_TRACKING_FRAME_THRESHOLD
        """
        self.get_logger().warn(f"Cannot retrieve target, so zooming out...")
        control_msg = GimbalControl()
        control_msg.msg_type = 1
        control_msg.zoom_in = 0
        self.control_publisher.publish(control_msg)



    # endregion

    # ============image related============
    # region image related
    def image_callback(self, msg):
        self.imageCount += 1
        if self.time_elapsed>1:
            if self.imageCount < self.fps:
                print(self.imageCount)
                print(self.time_elapsed)
                return
            if self.imageCount >= self.fps:
                self.time_elapsed=0
                self.imageCount = 1
                print("renew fps")
            
        

        start =time.time()
        # not run the model if not initialized
        if  self.state.is_intializing:
            return

        # not run the model if tracking is disabled
        if not self.state.enabled:
            return
        
        # wait gettting the gimbal info 
        if not self.init_gimbal_info:
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
        # tic1 = time.time()
        # print("after cv bridge:", tic1 -start  )

        #  apply_feature (GMC) will report the error when the feature is not enough 
        try:
            results , _ , _ = self.model.track(cv_image)
        except Exception as e :
            print("GMC error", e)
            return
        
        # tic2 =time.time()
        # print("after track:" , tic2 - tic1)
        # Prepare the objects' data

        num_detections = 0
        # analyze the results
        # if it is following object

        # track_time =time.time()
        # print("tracktime: " , track_time-readtime)
        self.objects_data = ObjectDetections()

        detected = False
        # if tracking ID is missing
        # print("length ", len(results))
        for t in results:
            self.detection = ObjectDetection()
            tlbr = t.tlbr
            tid = t.track_id
            tcls = t.cls
            c,  id = int(tcls), int(tid) % 65535
            x1,y1,x2,y2=tlbr[0],tlbr[1],tlbr[2],tlbr[3]


            self.detection.id = id  # Assign the detection ID
            self.detection.bbox = tlbr  # Replace with actual bbox coordinates
            self.detection.class_type = c  # Replace with actual class type
            self.detection.confidence = float(0)  # Replace with actual confidence
            num_detections +=1

            self.objects_data.detections.append(self.detection)


            # REID check (if enabled)
            if self.spatial_REID_enabled:
                if tid == self.currentID:
                    self.lastframe_istracking = 0
                    self.lastframe_istracking_after_zoom_out=0
                    detected = True

        if self.spatial_REID_enabled:
            # if we have not detected object and we had last frame
            if not detected and self.lastframe_istracking >= 3:
                # we try to find the one that match
                new_id = self.retrieve_target(results)
                if new_id is not None:
                    # publish the new id
                    self.ID_publisher.publish(Int32(data=new_id))
                    self.currentID = new_id
                    self.get_logger().warn(f"AUTO REID: New ID: {self.currentID}")
                    self.lastframe_istracking = 0
                else:
                    self.lastframe_istracking+=1

                    if self.lastframe_istracking > LOSE_TRACKING_FRAME_THRESHOLD: # which means the target has been lost for 5 frames
                        self.lastframe_istracking = -1
                        self.get_logger().warn(f"Cannot retrieve target {self.currentID}. Resetting tracking.")

                        # now we need to zoom out for a bit

                        self.zoom_out()
                        self.zooming_flag=True

            elif not detected and self.lastframe_istracking >=0 :
                self.lastframe_istracking+=1
            # for the search after zoom out 
            elif not detected and self.lastframe_istracking== -1 and self.zooming_flag:
                new_id = self.retrieve_target(results, zoom_flag=True)
                if new_id is not None:
                    # publish the new id
                    self.ID_publisher.publish(Int32(data=new_id))
                    self.currentID = new_id
                    self.get_logger().warn(f"AUTO REID: New ID: {self.currentID}")
                    self.lastframe_istracking = 0
                    self.lastframe_istracking_after_zoom_out=0
                    self.zooming_flag=False

                elif self.lastframe_istracking_after_zoom_out<LOSE_TRACKING_FRAME_THRESHOLD_AFTER_ZOOM:
                    self.lastframe_istracking_after_zoom_out+=1
                elif  self.lastframe_istracking_after_zoom_out>= LOSE_TRACKING_FRAME_THRESHOLD_AFTER_ZOOM:
                    self.get_logger().warn(f"Cannot retrieve target {self.currentID}. Resetting tracking.")
                    self.zooming_flag=False
                
        # tic3 =time.time()
        # print("after the reid save :", tic3 -tic2)
                

        #self.get_logger().info(f'object data: {objects_data}')
        #print(num_detections)
        if(num_detections>0): 
            self.objects_data.num_detections = num_detections
            self.objects_data.tracking_mode = 0 
            print("sent the detection msg")
            self.box_publisher.publish(self.objects_data)

        self.last_tracking_time = self.get_clock().now()
        self.state.is_detecting = True

        end =time.time()
        self.time_elapsed += (end - start)
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
