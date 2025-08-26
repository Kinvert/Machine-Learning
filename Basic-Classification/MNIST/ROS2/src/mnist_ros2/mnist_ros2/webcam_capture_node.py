import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import String

class WebcamCaptureNode(Node):
    def __init__(self):
        super().__init__('webcam_capture')
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)

        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)

        self.get_logger().info('Webcam node started. Press SPACE to publish frame, Q to quit')
        self.publish_frame = False

        self.prediction_sub = self.create_subscription(
            String, '/mnist_confidence', self.prediction_callback, 10
        )

        self.roi_size = 200
        self.current_prediction = "Draw a digit in the box"

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        half_roi = self.roi_size // 2

        x1 = center_x - half_roi
        y1 = center_y - half_roi
        x2 = center_x + half_roi
        y2 = center_y + half_roi

        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, "Put digit here", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display_frame, self.current_prediction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(display_frame, "SPACE: capture, Q: quit", (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Webcam - Press SPACE to capture, Q to quit', display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            roi = frame[y1:y2, x1:x2]
            ros_image = self.bridge.cv2_to_imgmsg(roi, "bgr8")
            self.image_pub.publish(ros_image)
            self.get_logger().info('ROI captured and published!')
        elif key == ord('q'):
            rclpy.shutdown()
            return
        
    def display_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        half_roi = self.roi_size // 2

        x1 = center_x - half_roi
        y1 = center_y - half_roi
        x2 = center_x + half_roi
        y2 = center_y + half_roi

        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, "Put digit here", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, self.current_prediction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, "SPACE: capture, Q: quit", (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Webcam - Press SPACE to capture, Q to quit', display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            roi = frame[y1:y2, x1:x2]
            ros_image = self.bridge.cv2_to_imgmsg(roi, "bgr8")
            self.image_pub.publish(ros_image)
            self.get_logger().info('ROI captured and published!')
        elif key == ord('q'):
            return False

        return True

    def prediction_callback(self, msg):
        self.current_prediction = msg.data

def main():
    rclpy.init()
    node = WebcamCaptureNode()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if not node.display_webcam():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
