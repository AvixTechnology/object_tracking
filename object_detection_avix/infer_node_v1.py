import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32MultiArray, Float32MultiArray, Int32
from ultralytics import YOLO
import torch
from object_detection_util import KalmanFilter
torch.cuda.device(0)
torch.manual_seed(23) #for cuda initialization bugs
torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        # Subscribe to necessary topics
        self.image_subscriber = self.create_subscription(Image, '/ktg_gimbal/image_raw', self.image_callback, 10)
        self.enable_subscriber = self.create_subscription(Bool, '/object_detection/tracking_enable', self.enable_callback, 10)
        self.target_subscriber = self.create_subscription(Int32, '/object_detection/tracking_target_id', self.target_id_callback, 10)
        self.target_object_subscriber = self.create_subscription(Int32MultiArray, '/object_detection/tracking_target_object', self.target_object_callback, 10)
        # Publisher for the deviation (???)
        self.deviation_publisher = self.create_publisher(Float32MultiArray, '/object_detection/target_deviation', 10)
        self.initializing = True
        # Initialize your YOLO model
        self.model =  YOLO('yolov8n.engine',task='detect')


        # initialize the parameter
        self.declare_parameter('tracking_enabled', False)
        self.tracking_enabled = self.get_parameter('tracking_enabled').value

        self.declare_parameter('following_ID', -1)
        self.following_id = self.get_parameter('following_ID').value

        self.declare_parameter('input_width', 640)
        self.input_width = self.get_parameter('input_width').value

        self.declare_parameter('input_height', 480)
        self.input_height = self.get_parameter('input_height').value

        self.initializing = False
        # node created
        self.get_logger().info(f'Object Detection Node created')




    def image_callback(self, msg):
        # Convert ROS Image message to CV2 format and pass it to your tracking system
        pass

    def target_object_callback(self,msg):
        pass

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
            pass

    def target_id_callback(self, msg):
        # Assuming msg.data contains the ID of the target to track
        self.tracking_system.startTracking(msg.data)

    def publish_deviation(self, deviation_x, deviation_y):
        msg = Float32MultiArray()
        msg.data = [deviation_x, deviation_y]
        self.deviation_publisher.publish(msg)