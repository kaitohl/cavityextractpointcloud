#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge
from skimage import filters, exposure, data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter, disk
from skimage import morphology

class SingleCaptureNode(Node):
    def __init__(self):
        super().__init__('single_capture_node')
        # Subscribe to the depth point cloud topic
        self.pc_subscription = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )
        # Subscribe to the color image topic
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        # Subscribe to the aligned depth image topic
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        self.capture_done = False
        self.depth_received = False
        self.points = None            # Will store the captured point cloud
        self.image_received = False
        self.cv_image = None          # Will store the converted color image
        self.bridge = CvBridge()

    def pointcloud_callback(self, msg):
        if self.capture_done:
            return
        self.get_logger().info("Received a pointcloud message!")
        # Convert the ROS2 PointCloud2 message to a list of (x, y, z) points
        points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        # Ensure each point is a list of floats
        points_list = [list(map(float, point)) for point in points_list]
        # Convert the list to an n x 3 NumPy array
        self.points = np.array(points_list, dtype=np.float32)
        self.capture_done = True
        self.get_logger().info("Point cloud capture complete.")

    def image_callback(self, msg):
        if self.image_received:
            return
        self.get_logger().info("Received an image message!")
        # Convert the ROS Image message to a CV2 image using cv_bridge.
        # Assuming the encoding is "rgb8"
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.image_received = True
        self.get_logger().info("Image capture complete.")

    def depth_callback(self, msg):
        if self.depth_received:
            return
        self.get_logger().info("Received a depth message!")
        self.cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_received = True  # Make sure the variable name is consistent.
        self.get_logger().info("Depth image capture complete.")

def main(args=None):
    print("Starting point cloud and image capture...")
    rclpy.init(args=args)
    node = SingleCaptureNode()
    node.get_logger().info("Waiting for point cloud and image...")

    # Spin until both point cloud and image have been received.
    while rclpy.ok() and (not node.capture_done or not node.image_received or not node.depth_received):
        rclpy.spin_once(node, timeout_sec=0.1)

    print("Both point cloud and image captured.")

    # Visualize the point cloud if available.
    if node.points is not None:
        points = node.points
        print("Visualizing captured point cloud...")
        # Filter points: for example, keep only those with z between 0 and 1
        points = points[(points[:, 2] > 0) & (points[:, 2] < 1)]
        # Subsample points to reduce the total number for visualization
        num_points = points.shape[0]
        num_points_to_sample = num_points // 8
        sampled_indices = np.random.choice(num_points, num_points_to_sample, replace=False)
        points = points[sampled_indices]
        # Create a 3D scatter plot
        # fig1 = plt.figure(figsize=(10, 10))
        # ax = fig1.add_subplot(111, projection='3d')
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('Captured Point Cloud')
    else:
        print("No point cloud captured.")

    if node.cv_image is not None:
        #selects the green channel of the image
        image = node.cv_image[:, :, 1]
        # Display the original image.
        fig2, ax2 = plt.subplots()
        ax2.imshow(image)
        ax2.set_title("Original Captured Image")
        ax2.axis('off')

        # Apply Canny edge detection
        edges = canny(image, sigma=3, low_threshold=5, high_threshold=20)
        fig3, ax3 = plt.subplots()
        ax3.imshow(edges, cmap='gray')
        ax3.set_title("Canny Edge Detection")
        ax3.axis('off')

        # Define a range of radii (you can adjust these values as needed)
        hough_radii = np.arange(70, 300, 2)

        # Perform Hough Circle Transform on the edge image
        hough_res = hough_circle(edges, hough_radii)

        # Extract the most prominent circle (total_num_peaks=1)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

        # For visualization, convert the grayscale image to RGB so you can overlay colored circles.
        # (If you want to overlay on the original RGB image and it is available, you can use that instead.)
        image_rgb = color.gray2rgb(image)

        # Draw the detected circle(s) on the image.
        for center_y, center_x, radius in zip(cy, cx, radii):
            # Optionally, you can adjust the radius (e.g., radius - 5) if needed.
            circy, circx = circle_perimeter(center_y, center_x, radius - 8, shape=image.shape)
            image_rgb[circy, circx] = (220, 20, 20)  # Use a red color for the circle outline

        # Plot the resulting image with the detected circle
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.imshow(image_rgb)
        ax4.set_title("Detected Circle")
        ax4.axis('off')
        plt.show()

        # create a mask for the detected circle, everything outside the circle is set to 0, everything inside is set to 1
        mask = np.zeros(image.shape, dtype=np.uint8)
        rr, cc = disk((cy[0], cx[0]), radii[0]-8, shape=image.shape)
        mask[rr, cc] = 1
        # plot the mask
        # fig5, ax5 = plt.subplots(figsize=(10, 4))
        # ax5.imshow(mask, cmap='gray')
        # ax5.set_title("Mask of Detected Circle")
        # ax5.axis('off')
        # plt.show()

    else:
        print("No image captured.")

    if node.cv_depth is not None:

        depthimage = node.cv_depth

        # Display the depth image.
        fig6, ax6 = plt.subplots()
        ax6.imshow(depthimage, cmap='gray')
        ax6.set_title("Depth Image")
        ax6.axis('off')
        plt.show()

        # Use the mask to filter the depth image
        masked_depth = np.zeros(depthimage.shape, dtype=np.float32)
        masked_depth[mask == 1] = depthimage[mask == 1]

        # Plot the masked depth image
        fig7, ax7 = plt.subplots()
        ax7.imshow(masked_depth)
        ax7.set_title("Masked Depth Image")
        ax7.axis('off')
        plt.show()

        # Calculate the maximum and minimum depth values in the masked depth image
        max_depth = np.max(masked_depth[:, :])
        print(max_depth)
        min_depth = np.average(masked_depth[:, :])
        print(min_depth)

        threshold = 25
        # plot masked_depth > (max_depth - threshold) points on top of image
        fig8, ax8 = plt.subplots()
        ax8.imshow(image, cmap='gray')
        ax8.set_title("Masked Depth Image")
        ax8.axis('off')
        ax8.imshow(masked_depth > (max_depth - threshold), cmap='hot', alpha=0.5)
        plt.show()

        # Create a mask for the points that satisfy the condition
        mask_condition = masked_depth > (max_depth - threshold)
        # Get the rows and columns of the points that satisfy the condition
        rows, cols = np.nonzero(mask_condition)
        depth_values = masked_depth[rows, cols]

        #reduce depth_values density by removing two points for every point
        depth_values = depth_values[::9]
        rows = rows[::9]
        cols = cols[::9]


        # Create a new 3D scatter plot
        # equal axis scaling
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create a scatter plot with the columns (x), rows (y), and depth (z).
        sc = ax.scatter(cols, rows, depth_values, c=depth_values, cmap='viridis', s=5)

        # Optionally invert the zaxis
        ax.invert_zaxis()

        # Add labels and a colorbar
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_zlabel('Depth')
        fig.colorbar(sc, ax=ax, label='Depth')

        plt.title("3D Scatter Plot of Nonzero Depth Values")
        plt.show()
    



    else:
        print("No depth image captured.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
