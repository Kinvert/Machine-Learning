import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MNISTClassifierNode(Node):
    def __init__(self):
        super().__init__('mnist_classifier')

        self.bridge = CvBridge()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN().to(self.device)

        current_dir = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'models', 'mnist_model.pth'
        )

        possible_paths = [
            os.path.join(current_dir, '..', '..', 'models', 'mnist_model.pth'),
            os.path.join(current_dir, '..', '..', '..', 'models', 'mnist_model.pth'),
            self.find_model_from_package_root()
        ]

        model_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                model_path = path
                break

        if not model_path:
            self.get_logger().error('Model not found! Searched in:')
            for path in possible_paths:
                if path:
                    self.get_logger().error(f'  {os.path.abspath(path)}')
            self.get_logger().error('Please run: cd src/mnist_ros2 && python3 train_mnist.py')
            return

        if model_path:
            #self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            self.get_logger().info(f'Model loaded from: {os.path.abspath(model_path)}')

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.prediction_pub = self.create_publisher(Int32, '/mnist_prediction', 10)
        self.confidence_pub = self.create_publisher(String, '/mnist_confidence', 10)
        self.processed_image_pub = self.create_publisher(Image, '/processed_image', 10)

        self.get_logger().info('MNIST Classifier Node started')
        self.get_logger().info('Subscribing to: /camera/image_raw')
        self.get_logger().info('Publishing to: /mnist_prediction, /mnist_confidence')

        self.latest_frame = None

    def find_model_from_package_root(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        search_dir = current_dir

        for _ in range(10):
            if os.path.exists(os.path.join(search_dir, 'package.xml')):
                model_path = os.path.join(search_dir, 'models', 'mnist_model.pth')
                return model_path
            search_dir = os.path.dirname(search_dir)
            if search_dir == '/':
                break
        return None

    def preprocess_image(self, cv_image):
        if len(cv_image.shape) == 3:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=-50)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return gray
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            processed = self.preprocess_image(cv_image)
            self.latest_frame = processed

            tensor_image = self.transform(processed).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_image)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                predicted_digit = predicted.item()
                confidence_score = confidence.item()

            prediction_msg = Int32()
            prediction_msg.data = predicted_digit
            self.prediction_pub.publish(prediction_msg)

            confidence_msg = String()
            confidence_msg.data = f"Digit: {predicted_digit}, Confidence: {confidence_score:.3f}"
            self.confidence_pub.publish(confidence_msg)

            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            processed_msg = self.bridge.cv2_to_imgmsg(processed_bgr, "bgr8")
            self.processed_image_pub.publish(processed_msg)

            self.get_logger().info(f'Predicted: {predicted_digit} (confidence: {confidence_score:.3f})')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    mnist_classifier = MNISTClassifierNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(mnist_classifier, timeout_sec=0.1)
            if mnist_classifier.latest_frame is not None:
                cv2.imshow("Result", mnist_classifier.latest_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        mnist_classifier.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
