import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32MultiArray, Float32MultiArray, Int32, ByteMultiArray
from ultralytics import YOLO
from avix_msg.msg import TrackingUpdate, ObjectDetection, ObjectDetections, Inf_info, Gimbal_info
import torch
import cv2
import struct
import math
from object_detection_avix.object_detection_util import KalmanFilter ,ReIDTrack
from cv_bridge import CvBridge, CvBridgeError
import numpy as np  


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
        self.enable_subscriber = self.create_subscription(Bool, '/object_detection/tracking_enable', self.enable_callback, 10)
        self.inf_subscriber = self.create_subscription(Inf_info, '/inf_interface/gps_info', self.gps_callback, 10)
        self.gimbal_subscriber = self.create_subscription(Gimbal_info, '/inf_interface/gps_info', self.gimbal_callback, 10)

        
        # Publisher for the deviation 
        self.deviation_publisher = self.create_publisher(TrackingUpdate, '/object_detection/target_deviation', 10)
        self.box_publisher = self.create_publisher(ObjectDetections, '/object_detection/detections', 10) # bytes array is not working yet
        self.initializing = True

        # initialize the parameter
        self.declare_parameter('tracking_enabled', False)
        #self.tracking_enabled = self.get_parameter('tracking_enabled').value
        self.tracking_enabled = True

        self.declare_parameter('confidence', 0.6)
        self.confidence = self.get_parameter('confidence').value

        self.declare_parameter('input_width', 640)
        self.input_width = self.get_parameter('input_width').value

        self.declare_parameter('input_height', 480)
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

        # Initialize your YOLO model
        self.get_logger().info(f'Model Initialized...')
        
        # Initialize CV bridge
        self.bridge = CvBridge()

        self.result=ReIDTrack()

        # initialize the tracking variable
        self.target_id = -1 # for following
        self.KF_initailized = False

        self.initializing = False
        self.ranging_flag = False # assume ranging is not available
        # node created
        self.detection = ObjectDetection()
        self.objects_data = ObjectDetections()
        self.get_logger().info(f'Object Detection Node created')


    def gps_callback(self, msg):
        self.gps_info = msg

        # update the require information
        self.longitude = msg.longitude
        self.lattitude = msg.lattitude
        self.altitude = msg.altitude

        self.uav_roll = msg.roll
        self.uav_pitch = msg.pitch
        self.uav_yaw = msg.yaw

    def gimbal_callback(self, msg):
        self.gimbal_info = msg

        # update the require information
        self.gimbal_pitch = msg.pitch_angle
        self.gimbal_yaw = msg.yaw_angle
        self.ranging_flag = msg.ranging_flag


    def find_location(self):
        if(self.ranging_flag):
            # Constants
            EARTH_RADIUS = 6371000  # in meters

            # Step 1: Calculate absolute gimbal orientation
            abs_gimbal_pitch = self.uav_pitch + self.gimbal_pitch
            abs_gimbal_yaw = self.uav_yaw + self.gimbal_yaw

            # Step 2: Calculate the distance to the object
            # Assuming camera is pointing directly downwards when gimbal_pitch is 0
            # Convert pitch angle from degrees to radians for trigonometric functions
            pitch_radians = math.radians(abs_gimbal_pitch)
            distance_to_object = self.altitude * math.tan(pitch_radians)

            # Step 3: Calculate the GPS location of the object
            # Convert yaw angle from degrees to radians
            yaw_radians = math.radians(abs_gimbal_yaw)

            # Calculate the displacement in the North and East directions
            delta_north = distance_to_object * math.cos(yaw_radians)
            delta_east = distance_to_object * math.sin(yaw_radians)

            # Convert displacement to degrees of latitude and longitude
            delta_latitude = (delta_north / EARTH_RADIUS) * (180 / math.pi)
            delta_longitude = (delta_east / EARTH_RADIUS) * (180 / math.pi) / math.cos(math.radians(self.latitude))

            # Calculate the final GPS location of the object
            object_latitude = self.latitude + delta_latitude
            object_longitude = self.longitude + delta_longitude
            object_altitude = 0
        else:
            # we use ranging data to find the location
            pass


        return object_latitude, object_longitude, object_altitude

        

    def image_callback(self, msg):
        # not run the model if not initialized
        if self.initializing:
            return

        # not run the model if tracking is disabled
        if not self.tracking_enabled:
            return
        
        # Convert ROS Image message to CV2 format and pass it to model
        try:
            # Convert the ROS Image message to a CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting ROS Image to OpenCV: {e}')
            return
        
        # Check the size of the image
        height, width = cv_image.shape[:2]
        if width != self.input_width or height != self.input_height:
            # resize the image
            #self.get_logger().warn(
              #  f"Received image of dimensions ({width}, {height}), which does not match expected dimensions ({self.input_width}, {self.input_height}). Resizing image.")
            cv_image = cv2.resize(cv_image, (self.input_width, self.input_height))

        # do the tracking
        
        results = self.result.track(cv_image)
        # Prepare the objects' data
        
        num_detections = 0
        # analyze the results
        # if it is following object
        
        
        
        
        
        
        for t in results:
            tlbr = t.tlbr
            tid = t.track_id
            tcls = t.cls
            c,  id = int(tcls), int(tid)
            x1,y1,x2,y2=tlbr[0],tlbr[1],tlbr[2],tlbr[3]
            if(id  == self.target_id):
                self.follow((x2+x1)/2,(y2+y1)/2)

                # check if it is inside the center of the image
                x_center = (x1+x2)/2
                y_center = (y1+y2)/2

                # check if it is within 30% around center
                if(x_center<self.input_width/2*1.3 and x_center>self.input_width/2*0.7 and y_center<self.input_height/2*1.3 and y_center>self.input_height/2*0.7):
                    #send the gps data
                    deduced_long, deduced_lat, deduced_alt = self.find_location() # next we need to use the angle to deduce the right one
            self.detection.id = id  # Assign the detection ID
            self.detection.bbox = tlbr  # Replace with actual bbox coordinates
            self.detection.class_type = c  # Replace with actual class type
            self.detection.confidence = float(0)  # Replace with actual confidence
            num_detections +=1
            self.objects_data.detections.append(self.detection)


        #self.get_logger().info(f'object data: {objects_data}')
        if(num_detections>0): 
            self.box_publisher.publish(self.objects_data)
            

    def follow(self, cx, cy):
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

    def enable_callback(self, msg):
        if self.initializing:
            self.get_logger().warn(f'Object Detection Model still initializing, please send command later')
            return
        self.set_parameter(rclpy.node.Parameter('tracking_enabled', rclpy.Parameter.Type.BOOL, msg.data))
        self.tracking_enabled = msg.data
        if msg.data:
            # enable tracking
            pass
        else:
            # disable tracking
            self.target_id = -1
            self.KF_initailized = False

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
