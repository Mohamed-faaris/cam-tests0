import cv2
import numpy as np
import time


# ============================================================
# CONFIGURATION
# ============================================================

BOARD_SIZE = (8, 5)

# Your physical checkerboard square size
SQUARE_SIZE_MM = 4.0

# Number of successful views required
REQUIRED_VIEWS = 25


# ============================================================
# CREATE 3D CHECKERBOARD POINTS
# ============================================================

# Example:
#
# (0,0,0)  (4,0,0)  (8,0,0) ...
#    ●-------●-------●
#    |
#    |
#    ●-------●-------●
#
# Everything is on Z = 0.

object_points_template = np.zeros(
    (BOARD_SIZE[0] * BOARD_SIZE[1], 3),
    np.float32
)

object_points_template[:, :2] = np.mgrid[
    0:BOARD_SIZE[0],
    0:BOARD_SIZE[1]
].T.reshape(-1, 2)

object_points_template *= SQUARE_SIZE_MM


# ============================================================
# STORAGE
# ============================================================

object_points = []
image_points = []


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# STATE
# ============================================================

captured = 0

print()
print("======================================")
print(" CAMERA CALIBRATION")
print("======================================")
print()
print("Board: 8 x 5 internal corners")
print("Square size: 4 mm")
print()
print("Move the checkerboard around.")
print("Press SPACE to capture a valid frame.")
print("Press ESC to exit.")
print()
print(f"Target: {REQUIRED_VIEWS} images")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera read error")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Find checkerboard
    # --------------------------------------------------------

    found, corners = cv2.findChessboardCornersSB(
        gray,
        BOARD_SIZE
    )

    # --------------------------------------------------------
    # Draw detection
    # --------------------------------------------------------

    if found:

        cv2.drawChessboardCorners(
            frame,
            BOARD_SIZE,
            corners,
            found
        )

        status = "BOARD FOUND"
        status_color = (0, 255, 0)

    else:

        status = "BOARD NOT FOUND"
        status_color = (0, 0, 255)

    # --------------------------------------------------------
    # Display status
    # --------------------------------------------------------

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    cv2.putText(
        frame,
        f"Captured: {captured}/{REQUIRED_VIEWS}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE = capture    ESC = exit",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Camera Calibration",
        frame
    )

    # --------------------------------------------------------
    # Keyboard
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # ESC
    if key == 27:
        break

    # SPACE
    if key == 32:

        if not found:

            print("Board not found - frame rejected")
            continue

        # ----------------------------------------------------
        # Refine corners
        # ----------------------------------------------------

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS
                + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        # ----------------------------------------------------
        # Store calibration data
        # ----------------------------------------------------

        object_points.append(
            object_points_template.copy()
        )

        image_points.append(
            corners_refined.copy()
        )

        captured += 1

        print(
            f"Captured {captured}/{REQUIRED_VIEWS}"
        )

        # ----------------------------------------------------
        # Finished?
        # ----------------------------------------------------

        if captured >= REQUIRED_VIEWS:

            print()
            print("Collected enough images.")
            print("Running calibration...")
            print()

            break


# ============================================================
# CLEANUP CAMERA
# ============================================================

cap.release()
cv2.destroyAllWindows()


# ============================================================
# CHECK DATA
# ============================================================

if captured < 10:

    raise RuntimeError(
        "Not enough calibration images."
    )


# ============================================================
# CALIBRATE
# ============================================================

image_size = gray.shape[::-1]

ret, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    image_size,
    None,
    None
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("======================================")
print(" CALIBRATION RESULT")
print("======================================")
print()

print("Camera matrix:")
print(camera_matrix)

print()

print("Distortion coefficients:")
print(distortion)

print()

print(f"RMS reprojection error: {ret:.6f} pixels")


# ============================================================
# SAVE
# ============================================================

np.savez(
    "camera_calibration.npz",
    camera_matrix=camera_matrix,
    distortion=distortion,
    image_width=image_size[0],
    image_height=image_size[1],
    square_size_mm=SQUARE_SIZE_MM,
    board_width=BOARD_SIZE[0],
    board_height=BOARD_SIZE[1]
)

print()
print("Saved:")
print("camera_calibration.npz")
print()