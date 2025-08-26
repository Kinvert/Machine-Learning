import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge
import cv2
import numpy as np

class DrawNumberNode(Node):
    def __init__(self):
        super().__init__('draw_number_node')
        self.bridge = CvBridge()
        self.canvas_size = (400, 400)
        self.canvas = np.zeros((*self.canvas_size, 3), dtype=np.uint8)
        self.drawing = False
        self.brush_size = 15

        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)

        self.prediction_sub = self.create_subscription(
            String,
            '/mnist_confidence',
            self.prediction_callback,
            10
        )

        cv2.namedWindow('Draw Digit Here')
        cv2.setMouseCallback('Draw Digit Here', self.mouse_callback)

        self.timer = self.create_timer(0.1, self.publish_image)  # 10 Hz

        self.get_logger().info('Draw Number Node started')
        self.get_logger().info('Draw digits in the window, press "c" to clear, "q" to quit')
        self.current_prediction = "No prediction yet"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.circle(self.canvas, (x, y), self.brush_size, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def prediction_callback(self, msg):
        self.current_prediction = msg.data

    def publish_image(self):
        display_canvas = self.canvas.copy()

        cv2.putText(display_canvas, self.current_prediction, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display_canvas, "Press 'c' to clear, 'q' to quit", 
                   (10, self.canvas_size[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Draw Digit Here', display_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            self.canvas = np.zeros((*self.canvas_size, 3), dtype=np.uint8)
            self.get_logger().info('Canvas cleared')
        elif key == ord('q'):
            self.get_logger().info('Quitting...')
            rclpy.shutdown()
            return

        try:
            ros_image = self.bridge.cv2_to_imgmsg(self.canvas, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f'Error publishing image: {str(e)}')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    draw_node = DrawNumberNode()

    try:
        rclpy.spin(draw_node)
    except KeyboardInterrupt:
        pass
    finally:
        draw_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
