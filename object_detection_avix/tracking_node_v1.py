import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32MultiArray, Float32MultiArray, Int32, ByteMultiArray
from ultralytics import YOLO
import torch
import cv2
import struct
from object_detection_avix.object_detection_util import KalmanFilter
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

def calc_crc(data):
    # Initialize the checksum to 0
    d =bytearray(data)
    checksum = 0
    # Iterate over the data and add each byte to the checksum
    for b in data:
        checksum += b

    checksum = checksum%65536 
     # Return the lower 2 bytes of the checksum as a 2-byte byte array compared to the crc
    return checksum.to_bytes(2, byteorder='little')

class TrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracking_node')
        # Subscribe to necessary topics
        self.image_subscriber = self.create_subscription(Image, '/ktg_gimbal/image_raw', self.image_callback, 10)
        self.enable_subscriber = self.create_subscription(Bool, '/object_detection/tracking_enable', self.enable_callback, 10)
        self.target_subscriber = self.create_subscription(Int32, '/object_detection/tracking_target_id', self.target_id_callback, 10)
        # Publisher for the deviation 
        self.deviation_publisher = self.create_publisher(Float32MultiArray, '/object_detection/target_deviation', 10)
        self.box_publisher = self.create_publisher(Int32MultiArray, '/object_detection/packet_data', 10) # bytes array is not working yet
        self.initializing = True
        
        #check if enviornment is ok
        if(not torch.cuda.is_available()):
            self.get_logger().warn(f'Pytorch has no cuda support, please reinstall')

        # Get the directory where your package is installed
        package_dir = get_package_share_directory('object_detection_avix')
        
        # Construct the full path to the .engine file
        engine_path = os.path.join(package_dir, 'yolov8n.engine')

        self.get_logger().info(f'Initializing Model...')

        # Initialize your YOLO model
        self.model =  YOLO(engine_path,task='detect')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # initialize the parameter
        self.declare_parameter('tracking_enabled', False)
        #self.tracking_enabled = self.get_parameter('tracking_enabled').value
        self.tracking_enabled = True

        self.declare_parameter('following_ID', -1)
        self.following_id = self.get_parameter('following_ID').value

        self.declare_parameter('confidence', 0.6)
        self.confidence = self.get_parameter('confidence').value

        self.declare_parameter('input_width', 640)
        self.input_width = self.get_parameter('input_width').value

        self.declare_parameter('input_height', 480)
        self.input_height = self.get_parameter('input_height').value

        self.declare_parameter('target_object', [0])
        self.target_objects = self.get_parameter('target_object').value

        # initialize the tracking variable
        self.target_id = -1 # for following
        self.KF_initailized = False

        self.initializing = False
        # node created
        self.get_logger().info(f'Object Detection Node created')




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
            self.get_logger().warn(
                f"Received image of dimensions ({width}, {height}), which does not match expected dimensions ({self.input_width}, {self.input_height}). Resizing image.")
            cv_image = cv2.resize(cv_image, (self.input_width, self.input_height))

        # do the tracking
        results = self.model.track(cv_image, persist=True, conf=self.confidence, classes=self.target_objects, verbose=False, imgsz=(self.input_height,self.input_width))
        # Prepare the objects' data
        objects_data = b''
        # analyze the results
        # if it is following object
        if(self.following_id >= 0):
             for d in reversed(results[0].boxes):
                c = int(d.cls)
                obj_id = 0 if d.id is None else int(d.id.item())
                xyxy = d.xyxy.squeeze()
                # Convert coordinates to the required format and pack them into bytes
                x1, y1, x2, y2 = [int(coord * 10) for coord in xyxy]
                if(obj_id == self.following_id):
                    self.follow((x2+x1)/20,(y2+y1)/20) #20 because it is multiplied by 10
                    c = 99 # object we are tracking
                obj_data = struct.pack('<HHHHHB', x1, y1, x2, y2, obj_id, c)
                objects_data += obj_data
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                name = ('' if id is None else f'id:{id} ') + results[0].names[c]
                label = (f'{name} {conf:.2f}' if conf else name) 

        else:
             for d in reversed(results[0].boxes):
                c = int(d.cls)
                obj_id = 0 if d.id is None else int(d.id.item())
                xyxy = d.xyxy.squeeze()
                
                # Convert coordinates to the required format and pack them into bytes
                x1, y1, x2, y2 = [int(coord * 10) for coord in xyxy]
                obj_data = struct.pack('<HHHHHB', x1, y1, x2, y2, obj_id, c)
                objects_data += obj_data
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                name = ('' if id is None else f'id:{id} ') + results[0].names[c]
                label = (f'{name} {conf:.2f}' if conf else name) 
        #self.get_logger().info(f'object data: {objects_data}')
        if(objects_data): 
            length_byte = struct.pack('B', (len(sender_id) + len(objects_data)))
            packet = packet_header + length_byte + sender_id + objects_data
            crc = calc_crc(packet)
            packet += crc
            packet_data =  [int(byte) for byte in packet]
            # self.get_logger().info(f'packet data: {packet_data}')
            # Create a ByteMultiArray message and assign the data
            self.box_publisher.publish(Int32MultiArray(data = packet_data))
            

    def follow(self, cx, cy):
        if(not self.KF_initailized):
            self.initialize_KF(cx,cy)

        filtered_x = np.dot(self.H,  self.KF_x.predict())[0]
        filtered_y = np.dot(self.H,  self.KF_y.predict())[0]

        # publish the deviation
        self.deviation_publisher.publish(Float32MultiArray(data = [filtered_x, filtered_y]))

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
            self.following_id = -1
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