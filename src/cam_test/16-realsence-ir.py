import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

# RGB
config.enable_stream(
    rs.stream.color,
    1280, 720,
    rs.format.bgr8,
    30
)

# IR 1
config.enable_stream(
    rs.stream.infrared,
    1,
    1280, 720,
    rs.format.y8,
    30
)

# IR 2
config.enable_stream(
    rs.stream.infrared,
    2,
    1280, 720,
    rs.format.y8,
    30
)

# DEPTH
config.enable_stream(
    rs.stream.depth,
    1280, 720,
    rs.format.z16,
    30
)

# Start
profile = pipeline.start(config)

# IR emitter control
device = profile.get_device()
sensor = device.first_depth_sensor()

emitter_on = True

if sensor.supports(rs.option.emitter_enabled):
    sensor.set_option(
        rs.option.emitter_enabled,
        1
    )

# Depth colorizer
colorizer = rs.colorizer()

# Window
window_name = "Intel RealSense D415"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

try:

    while True:

        frames = pipeline.wait_for_frames()

        # Get frames
        color_frame = frames.get_color_frame()
        ir1_frame = frames.get_infrared_frame(1)
        ir2_frame = frames.get_infrared_frame(2)
        depth_frame = frames.get_depth_frame()

        if not color_frame:
            continue

        if not ir1_frame:
            continue

        if not ir2_frame:
            continue

        if not depth_frame:
            continue

        # -------------------------
        # Convert frames
        # -------------------------

        color = np.asanyarray(
            color_frame.get_data()
        )

        ir1 = np.asanyarray(
            ir1_frame.get_data()
        )

        ir2 = np.asanyarray(
            ir2_frame.get_data()
        )

        # Colorized depth
        depth_color_frame = colorizer.colorize(depth_frame)

        depth = np.asanyarray(
            depth_color_frame.get_data()
        )

        # -------------------------
        # Resize
        # -------------------------

        size = (640, 360)

        color = cv2.resize(
            color,
            size
        )

        ir1 = cv2.resize(
            ir1,
            size
        )

        ir2 = cv2.resize(
            ir2,
            size
        )

        depth = cv2.resize(
            depth,
            size
        )

        # -------------------------
        # IR grayscale -> BGR
        # -------------------------

        ir1 = cv2.cvtColor(
            ir1,
            cv2.COLOR_GRAY2BGR
        )

        ir2 = cv2.cvtColor(
            ir2,
            cv2.COLOR_GRAY2BGR
        )

        # -------------------------
        # Labels
        # -------------------------

        cv2.putText(
            color,
            "RGB",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            ir1,
            "IR 1",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            ir2,
            "IR 2",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            depth,
            "DEPTH",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # -------------------------
        # 2 x 2 layout
        # -------------------------

        top = np.hstack(
            (color, ir1)
        )

        bottom = np.hstack(
            (ir2, depth)
        )

        output = np.vstack(
            (top, bottom)
        )

        cv2.imshow(
            window_name,
            output
        )

        # -------------------------
        # Keyboard
        # -------------------------

        key = cv2.waitKey(1) & 0xFF

        # Q / ESC = quit
        if key == ord('q') or key == 27:
            break

        # E = emitter ON/OFF
        if key == ord('e'):

            emitter_on = not emitter_on

            if sensor.supports(
                rs.option.emitter_enabled
            ):
                sensor.set_option(
                    rs.option.emitter_enabled,
                    1 if emitter_on else 0
                )

            print(
                "IR emitter:",
                "ON" if emitter_on else "OFF"
            )

finally:

    pipeline.stop()
    cv2.destroyAllWindows()