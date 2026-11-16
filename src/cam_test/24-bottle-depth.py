import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# COCO bottle class
BOTTLE_CLASS = 39

# Confidence threshold
CONFIDENCE = 0.5

# Depth is sampled from the CENTER of the bounding box.
#
# Using only the exact center pixel is noisy, so we take the
# median of the middle region instead.
#
# 0.5 = middle 50% of the box

DEPTH_SAMPLE_SCALE = 0.5


# ============================================================
# YOLO SETUP
# ============================================================

model = YOLO("yolo26m.pt")


# ============================================================
# DEPTH SAMPLING
# ============================================================

def sample_depth_mm(
    depth_image,
    x1,
    y1,
    x2,
    y2
):
    """Median depth of the center of a box, in mm."""

    height, width = depth_image.shape

    # Clamp box to image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return None

    # Shrink to center region
    shrink_x = int(
        box_w * (1.0 - DEPTH_SAMPLE_SCALE) / 2.0
    )

    shrink_y = int(
        box_h * (1.0 - DEPTH_SAMPLE_SCALE) / 2.0
    )

    cx1 = x1 + shrink_x
    cy1 = y1 + shrink_y
    cx2 = x2 - shrink_x
    cy2 = y2 - shrink_y

    region = depth_image[
        cy1:cy2,
        cx1:cx2
    ]

    if region.size == 0:
        return None

    # 0 = invalid depth (holes)
    valid = region[region > 0]

    if valid.size == 0:
        return None

    median_depth_m = float(
        np.median(valid)
    )

    return median_depth_m * 1000.0


# ============================================================
# REALSENSE SETUP
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

# RGB
config.enable_stream(
    rs.stream.color,
    1280, 720,
    rs.format.bgr8,
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

# Depth scale (usually 0.001 -> mm)
depth_sensor = (
    profile.get_device()
    .first_depth_sensor()
)

depth_scale = depth_sensor.get_depth_scale()

print()
print(
    f"Depth scale: {depth_scale}"
)

# IR emitter control
emitter_on = True

if depth_sensor.supports(
    rs.option.emitter_enabled
):
    depth_sensor.set_option(
        rs.option.emitter_enabled,
        1
    )

# Align depth to color so the
# YOLO box pixels line up with
# the depth pixels
align = rs.align(rs.stream.color)

# Depth colorizer
colorizer = rs.colorizer()

# Window
window_name = "Bottle Depth"
cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)


try:

    while True:

        frames = pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = align.process(
            frames
        )

        # Get frames
        color_frame = (
            aligned_frames.get_color_frame()
        )

        depth_frame = (
            aligned_frames.get_depth_frame()
        )

        if not color_frame:
            continue

        if not depth_frame:
            continue

        # -------------------------
        # Convert frames
        # -------------------------

        color = np.asanyarray(
            color_frame.get_data()
        )

        # Raw depth in units of
        # depth_scale (mm)
        depth_raw = np.asanyarray(
            depth_frame.get_data()
        )

        # Real depth in meters
        depth_m = (
            depth_raw.astype(np.float64)
            * depth_scale
        )

        # Colorized depth
        depth_color_frame = (
            colorizer.colorize(depth_frame)
        )

        depth_view = np.asanyarray(
            depth_color_frame.get_data()
        )

        # -------------------------
        # YOLO bottle detection
        # -------------------------

        results = model(
            color,
            classes=[BOTTLE_CLASS],
            conf=CONFIDENCE,
            verbose=False
        )

        detections = []

        boxes = (
            results[0].boxes
        )

        if boxes is not None:

            for box in boxes:

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                x1, y1, x2, y2 = xyxy

                depth_mm = sample_depth_mm(
                    depth_m,
                    x1, y1, x2, y2
                )

                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "depth_mm": depth_mm
                })

        # -------------------------
        # Print depths
        # -------------------------

        if len(detections) > 0:

            print(
                f"Bottles: {len(detections)}"
            )

            for i, det in enumerate(
                detections,
                start=1
            ):

                depth_mm = det["depth_mm"]

                if depth_mm is None:

                    print(
                        f"  {i}: depth ERROR"
                    )

                else:

                    print(
                        f"  {i}: "
                        f"{depth_mm:.0f} mm"
                    )

        # -------------------------
        # Draw boxes
        # -------------------------

        output = color.copy()

        for i, det in enumerate(
            detections,
            start=1
        ):

            x1 = det["x1"]
            y1 = det["y1"]
            x2 = det["x2"]
            y2 = det["y2"]

            # Box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            depth_mm = det["depth_mm"]

            if depth_mm is None:

                label = f"{i} bottle ERROR"

            else:

                label = (
                    f"{i} bottle "
                    f"{depth_mm:.0f} mm"
                )

            cv2.putText(
                output,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # -------------------------
        # Labels
        # -------------------------

        cv2.putText(
            output,
            "RGB + BOTTLE DEPTH",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            depth_view,
            "DEPTH",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # -------------------------
        # Legend
        # -------------------------

        legend_lines = [
            (
                "LEGEND",
                (255, 255, 255)
            ),
            (
                "Green box = bottle",
                (0, 255, 0)
            ),
            (
                "Red text = distance (mm)",
                (0, 0, 255)
            ),
            (
                f"Bottles: {len(detections)}",
                (255, 255, 255)
            ),
            (
                "Q = quit   E = emitter",
                (255, 255, 255)
            )
        ]

        # -------------------------
        # IR emitter status
        # -------------------------

        if emitter_on:

            ir_text = "IR emitter: ON"
            ir_color = (0, 255, 0)

        else:

            ir_text = "IR emitter: OFF"
            ir_color = (0, 0, 255)

        legend_lines.append(
            (ir_text, ir_color)
        )

        # -------------------------
        # Draw legend box
        # -------------------------

        legend_x = 20
        legend_y = 70
        line_height = 28

        legend_h = (
            len(legend_lines) * line_height
            + 20
        )

        legend_w = 420

        overlay = output.copy()

        cv2.rectangle(
            overlay,
            (legend_x - 10, legend_y - 30),
            (
                legend_x + legend_w,
                legend_y + legend_h
            ),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.5,
            output,
            0.5,
            0,
            output
        )

        for j, (
            text,
            color
        ) in enumerate(legend_lines):

            y = (
                legend_y
                + j * line_height
            )

            cv2.putText(
                output,
                text,
                (legend_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # -------------------------
        # Side by side
        # -------------------------

        side = np.hstack(
            (output, depth_view)
        )

        cv2.imshow(
            window_name,
            side
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

            if depth_sensor.supports(
                rs.option.emitter_enabled
            ):
                depth_sensor.set_option(
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
