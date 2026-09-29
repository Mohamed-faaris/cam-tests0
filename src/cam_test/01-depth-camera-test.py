import pyrealsense2 as rs
import numpy as np
import cv2

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for synchronized depth + color frames
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to NumPy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert depth to a displayable color image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Make dimensions match
        if depth_colormap.shape != color_image.shape:
            color_image = cv2.resize(
                color_image,
                (depth_colormap.shape[1], depth_colormap.shape[0]),
                interpolation=cv2.INTER_AREA
            )

        # Place RGB and depth side-by-side
        combined = np.hstack((color_image, depth_colormap))

        # Display
        cv2.imshow("RealSense RGB + Depth", combined)

        # Press Q or ESC to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()