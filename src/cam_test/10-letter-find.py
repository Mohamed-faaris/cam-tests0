import cv2
import numpy as np
import pytesseract
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# logo-find.py
#     ↓ parents[0] = cam_test
#     ↓ parents[1] = src
#     ↓ parents[2] = cam-test project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIBRATION_PATH = PROJECT_ROOT / "camera_calibration.npz"


# ============================================================
# CONFIG
# ============================================================

CAMERA_ID = 0

# Checkerboard:
# 9 x 6 checkerboard squares
# therefore 8 x 5 internal corners

BOARD_SIZE = (8, 5)

# Physical checkerboard square size
SQUARE_SIZE_MM = 40.0


# ============================================================
# OCR CONFIGURATION
# ============================================================

# The only characters we care about.
#
# These come from:
#
#     vMeasure
#     PALLET
#     ULTIMA
#
# Unique characters:
#
#     V M E A S U R P L T I
#
# We use uppercase because OCR output is converted to uppercase.

TARGET_CHARS = set(
    "VMEASURPLTI"
)

# Tesseract whitelist.
#
# OCR is told to look primarily for these characters.
OCR_WHITELIST = "VMEASURPLTI"


# Default OCR confidence.
#
# You can change this live using the slider.
DEFAULT_CONFIDENCE = 35


# ============================================================
# CHECK TESSERACT
# ============================================================

try:
    # If Tesseract is installed and available in PATH,
    # pytesseract will find it automatically.
    tesseract_version = pytesseract.get_tesseract_version()

    print("=" * 60)
    print("TESSERACT OCR")
    print("=" * 60)
    print(f"Version: {tesseract_version}")

except Exception as e:

    raise RuntimeError(
        "\nTesseract OCR was not found.\n\n"
        "Install Tesseract first.\n"
        "Then install pytesseract with:\n\n"
        "    pip install pytesseract\n\n"
        "On Windows, install Tesseract OCR and make sure "
        "tesseract.exe is available in PATH.\n"
    ) from e


# ============================================================
# CHECK CALIBRATION FILE
# ============================================================

if not CALIBRATION_PATH.exists():

    raise FileNotFoundError(
        f"Calibration not found:\n{CALIBRATION_PATH}"
    )


# ============================================================
# START
# ============================================================

print("=" * 60)
print("TEXT / CHARACTER POSITION TEST")
print("=" * 60)

print(f"Project root: {PROJECT_ROOT}")
print(f"Calibration:  {CALIBRATION_PATH}")

print()
print("Target text:")
print("    vMeasure PALLET ULTIMA")
print()
print("Detection rule:")
print("    ANY ONE TARGET CHARACTER IS ENOUGH")
print()


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

calibration = np.load(
    str(CALIBRATION_PATH)
)

camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion"]


print("Camera matrix:")
print(camera_matrix)

print("\nDistortion:")
print(distortion)


# ============================================================
# CHECKERBOARD 3D COORDINATES
# ============================================================

#
# Coordinate system:
#
#             Y
#             ↑
#             |
#             |
#             O ─────────→ X
#
# Z = 0 is the checkerboard plane
#

board_points = np.zeros(
    (
        BOARD_SIZE[0] * BOARD_SIZE[1],
        3
    ),
    dtype=np.float32
)


board_points[:, :2] = np.mgrid[
    0:BOARD_SIZE[0],
    0:BOARD_SIZE[1]
].T.reshape(-1, 2)


board_points *= SQUARE_SIZE_MM


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open camera."
    )


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# OCR WINDOW
# ============================================================

WINDOW_NAME = "Text Position"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


# ============================================================
# OCR CONFIDENCE SLIDER
# ============================================================

cv2.createTrackbar(
    "OCR Confidence",
    WINDOW_NAME,
    DEFAULT_CONFIDENCE,
    100,
    lambda value: None
)


# ============================================================
# PIXEL -> WORLD COORDINATE
# ============================================================

def pixel_to_world(
    pixel,
    rvec,
    tvec
):
    """
    Convert a camera pixel coordinate into
    checkerboard/world coordinates.

    Assumes the detected point lies on:

        Z = 0

    checkerboard plane.
    """

    # --------------------------------------------------------
    # Rotation matrix
    # --------------------------------------------------------

    R, _ = cv2.Rodrigues(
        rvec
    )


    # --------------------------------------------------------
    # Undistort pixel
    # --------------------------------------------------------

    pixel_array = np.array(
        [[
            [
                float(pixel[0]),
                float(pixel[1])
            ]
        ]],
        dtype=np.float64
    )


    undistorted = cv2.undistortPoints(
        pixel_array,
        camera_matrix,
        distortion
    )


    x = float(
        undistorted[0, 0, 0]
    )

    y = float(
        undistorted[0, 0, 1]
    )


    # --------------------------------------------------------
    # Ray in camera coordinates
    # --------------------------------------------------------

    ray_camera = np.array(
        [
            x,
            y,
            1.0
        ],
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Camera position in world coordinates
    # --------------------------------------------------------

    R_inv = R.T

    camera_position_world = (
        -R_inv @ tvec.reshape(3)
    )


    # --------------------------------------------------------
    # Ray direction in world coordinates
    # --------------------------------------------------------

    ray_world = (
        R_inv @ ray_camera
    )


    # --------------------------------------------------------
    # Intersect ray with Z = 0
    # --------------------------------------------------------

    if abs(ray_world[2]) < 1e-10:

        return None


    scale = (
        -camera_position_world[2]
        / ray_world[2]
    )


    # If scale is negative, the intersection
    # is behind the camera.
    if scale < 0:

        return None


    world_point = (
        camera_position_world
        + scale * ray_world
    )


    return world_point


# ============================================================
# OCR DETECTION
# ============================================================

def detect_target_character(
    frame,
    confidence_threshold
):
    """
    Detect any character belonging to:

        VMEASURE PALLET ULTIMA

    Returns:

        detected
        center_pixel
        detected_character
        confidence
        box
        ocr_text
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Improve OCR image
    # --------------------------------------------------------

    # Upscaling makes small letters easier for OCR.
    scale = 2.0

    enlarged = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


    # Mild contrast enhancement
    enhanced = cv2.equalizeHist(
        enlarged
    )


    # --------------------------------------------------------
    # Tesseract configuration
    # --------------------------------------------------------

    config = (
        "--psm 11 "
        "-c tessedit_char_whitelist="
        + OCR_WHITELIST
    )


    # --------------------------------------------------------
    # OCR with bounding boxes
    # --------------------------------------------------------

    data = pytesseract.image_to_data(
        enhanced,
        config=config,
        output_type=pytesseract.Output.DICT
    )


    best_detection = None

    detected_text_parts = []


    # --------------------------------------------------------
    # Examine OCR results
    # --------------------------------------------------------

    for i in range(
        len(data["text"])
    ):

        raw_text = data["text"][i].strip()

        if not raw_text:
            continue


        # OCR confidence
        try:
            confidence = float(
                data["conf"][i]
            )
        except Exception:
            continue


        if confidence < 0:
            continue


        # Save OCR text for debugging
        detected_text_parts.append(
            raw_text
        )


        # ----------------------------------------------------
        # Convert OCR text to uppercase
        # ----------------------------------------------------

        text = raw_text.upper()


        # ----------------------------------------------------
        # Check every character
        #
        # ANY ONE target character is enough.
        # ----------------------------------------------------

        valid_characters = []

        for character in text:

            if character in TARGET_CHARS:

                valid_characters.append(
                    character
                )


        if not valid_characters:
            continue


        # ----------------------------------------------------
        # Confidence threshold
        # ----------------------------------------------------

        if confidence < confidence_threshold:

            continue


        # ----------------------------------------------------
        # Bounding box
        #
        # Tesseract coordinates are based on the
        # enlarged image, so divide by scale.
        # ----------------------------------------------------

        x = int(
            data["left"][i] / scale
        )

        y = int(
            data["top"][i] / scale
        )

        w = int(
            data["width"][i] / scale
        )

        h = int(
            data["height"][i] / scale
        )


        # ----------------------------------------------------
        # Center of detected text
        # ----------------------------------------------------

        cx = int(
            round(x + w / 2.0)
        )

        cy = int(
            round(y + h / 2.0)
        )


        candidate = {
            "character": valid_characters[0],
            "confidence": confidence,
            "center": (cx, cy),
            "box": (x, y, w, h),
            "text": raw_text
        }


        # ----------------------------------------------------
        # Keep highest-confidence detection
        # ----------------------------------------------------

        if (
            best_detection is None
            or confidence
            > best_detection["confidence"]
        ):

            best_detection = candidate


    # --------------------------------------------------------
    # Return no detection
    # --------------------------------------------------------

    if best_detection is None:

        return (
            False,
            None,
            None,
            0.0,
            None,
            " ".join(detected_text_parts)
        )


    # --------------------------------------------------------
    # Return best detection
    # --------------------------------------------------------

    return (
        True,
        best_detection["center"],
        best_detection["character"],
        best_detection["confidence"],
        best_detection["box"],
        " ".join(detected_text_parts)
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # READ CAMERA
    # ========================================================

    ret, frame = cap.read()


    if not ret:

        print("\nCamera read failed.")

        break


    display = frame.copy()


    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # OCR CONFIDENCE SLIDER
    # ========================================================

    confidence_threshold = cv2.getTrackbarPos(
        "OCR Confidence",
        WINDOW_NAME
    )


    # ========================================================
    # 1. FIND CHECKERBOARD
    # ========================================================

    board_found, corners = (
        cv2.findChessboardCornersSB(
            gray,
            BOARD_SIZE
        )
    )


    rvec = None
    tvec = None


    # ========================================================
    # CHECKERBOARD POSE
    # ========================================================

    if board_found:

        # ----------------------------------------------------
        # Estimate checkerboard pose
        # ----------------------------------------------------

        success, rvec, tvec = cv2.solvePnP(
            board_points,
            corners,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )


        if success:

            # ------------------------------------------------
            # Draw checkerboard
            # ------------------------------------------------

            cv2.drawChessboardCorners(
                display,
                BOARD_SIZE,
                corners,
                board_found
            )


            # =================================================
            # DRAW WORLD AXES
            # =================================================

            axis_length = 40.0


            axis_points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [axis_length, 0.0, 0.0],
                    [0.0, axis_length, 0.0],
                    [0.0, 0.0, -axis_length]
                ],
                dtype=np.float32
            )


            projected, _ = cv2.projectPoints(
                axis_points,
                rvec,
                tvec,
                camera_matrix,
                distortion
            )


            projected = projected.reshape(
                -1,
                2
            )


            # ------------------------------------------------
            # Convert numpy floats to Python ints
            # ------------------------------------------------

            origin = (
                int(round(projected[0][0])),
                int(round(projected[0][1]))
            )


            x_axis = (
                int(round(projected[1][0])),
                int(round(projected[1][1]))
            )


            y_axis = (
                int(round(projected[2][0])),
                int(round(projected[2][1]))
            )


            z_axis = (
                int(round(projected[3][0])),
                int(round(projected[3][1]))
            )


            # ------------------------------------------------
            # X axis - red
            # ------------------------------------------------

            cv2.line(
                display,
                origin,
                x_axis,
                (0, 0, 255),
                3
            )


            # ------------------------------------------------
            # Y axis - green
            # ------------------------------------------------

            cv2.line(
                display,
                origin,
                y_axis,
                (0, 255, 0),
                3
            )


            # ------------------------------------------------
            # Z axis - blue
            # ------------------------------------------------

            cv2.line(
                display,
                origin,
                z_axis,
                (255, 0, 0),
                3
            )


            # ------------------------------------------------
            # Origin
            # ------------------------------------------------

            cv2.circle(
                display,
                origin,
                7,
                (255, 255, 255),
                -1
            )


            cv2.putText(
                display,
                "ORIGIN",
                (
                    origin[0] + 10,
                    origin[1]
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )


    # ========================================================
    # 2. OCR TARGET CHARACTER DETECTION
    # ========================================================

    (
        text_detected,
        text_center_pixel,
        detected_character,
        text_confidence,
        text_box,
        ocr_text
    ) = detect_target_character(
        frame,
        confidence_threshold
    )


    # ========================================================
    # DRAW OCR DETECTION
    # ========================================================

    if text_detected:

        x, y, w, h = text_box


        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )


        # ----------------------------------------------------
        # Draw center
        # ----------------------------------------------------

        cv2.circle(
            display,
            text_center_pixel,
            8,
            (0, 255, 255),
            -1
        )


        # ----------------------------------------------------
        # Detection label
        # ----------------------------------------------------

        label = (
            f"FOUND: {detected_character} "
            f"{text_confidence:.0f}%"
        )


        cv2.putText(
            display,
            label,
            (
                x,
                max(30, y - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ========================================================
    # 3. CONVERT CHARACTER CENTER TO BOARD COORDINATES
    # ========================================================

    world = None


    if (
        text_detected
        and rvec is not None
        and tvec is not None
    ):

        world = pixel_to_world(
            text_center_pixel,
            rvec,
            tvec
        )


        if world is not None:

            X = float(world[0])
            Y = float(world[1])
            Z = float(world[2])


            # ------------------------------------------------
            # Distance from checkerboard origin
            # ------------------------------------------------

            distance = np.sqrt(
                X * X +
                Y * Y
            )


            # =================================================
            # DISPLAY X
            # =================================================

            cv2.putText(
                display,
                f"X = {X:.2f} mm",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


            # =================================================
            # DISPLAY Y
            # =================================================

            cv2.putText(
                display,
                f"Y = {Y:.2f} mm",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


            # =================================================
            # DISPLAY Z
            # =================================================

            cv2.putText(
                display,
                f"Z = {Z:.2f} mm",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


            # =================================================
            # DISPLAY DISTANCE
            # =================================================

            cv2.putText(
                display,
                f"Distance = {distance:.2f} mm",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


            # =================================================
            # DISPLAY DETECTED CHARACTER
            # =================================================

            cv2.putText(
                display,
                f"Character = {detected_character}",
                (20, 255),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            # =================================================
            # CONSOLE OUTPUT
            # =================================================

            print(
                "\r"
                f"CHARACTER FOUND | "
                f"{detected_character} | "
                f"X={X:8.2f} mm | "
                f"Y={Y:8.2f} mm | "
                f"Z={Z:8.2f} mm | "
                f"D={distance:8.2f} mm | "
                f"conf={text_confidence:5.1f}%",
                end="",
                flush=True
            )


    # ========================================================
    # STATUS
    # ========================================================

    if not board_found:

        cv2.putText(
            display,
            "CHECKERBOARD NOT FOUND",
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    elif not text_detected:

        cv2.putText(
            display,
            "TARGET CHARACTER NOT FOUND",
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    elif world is None:

        cv2.putText(
            display,
            "CHARACTER FOUND - WORLD POSITION FAILED",
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )


    else:

        cv2.putText(
            display,
            "TARGET CHARACTER FOUND",
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ========================================================
    # OCR INFORMATION
    # ========================================================

    cv2.putText(
        display,
        f"Confidence threshold: {confidence_threshold}%",
        (20, 335),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # TARGET TEXT
    # ========================================================

    cv2.putText(
        display,
        "Target: vMeasure PALLET ULTIMA",
        (20, 365),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        display,
        "ESC = exit",
        (
            20,
            display.shape[0] - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\n")
print("=" * 60)
print("EXIT")
print("=" * 60)