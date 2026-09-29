import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BOARD_SIZE = (8, 5)       # 9x6 squares -> 8x5 internal corners
SQUARE_SIZE_MM = 40.0 # Size of a square in mm


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

data = np.load("camera_calibration.npz")

camera_matrix = data["camera_matrix"]
distortion = data["distortion"]


# ============================================================
# CHECKERBOARD 3D POINTS
#
# Coordinate system:
#
#          Y
#          ^
#          |
#          |
#          O -------> X
#
# Z = 0
#
# ============================================================

object_points = np.zeros(
    (BOARD_SIZE[0] * BOARD_SIZE[1], 3),
    dtype=np.float32
)

object_points[:, :2] = np.mgrid[
    0:BOARD_SIZE[0],
    0:BOARD_SIZE[1]
].T.reshape(-1, 2)

object_points *= SQUARE_SIZE_MM


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# GLOBALS
# ============================================================

selected_pixel = None
selected_world = None


# ============================================================
# PIXEL -> WORLD COORDINATE
#
# The object is assumed to be on Z = 0.
#
# ============================================================

def pixel_to_world(
    pixel,
    rvec,
    tvec
):

    # --------------------------------------------------------
    # Convert rotation vector to rotation matrix
    # --------------------------------------------------------

    R, _ = cv2.Rodrigues(rvec)

    # --------------------------------------------------------
    # Pixel -> normalized camera ray
    # --------------------------------------------------------

    pixel_array = np.array(
        [[[pixel[0], pixel[1]]]],
        dtype=np.float64
    )

    undistorted = cv2.undistortPoints(
        pixel_array,
        camera_matrix,
        distortion
    )

    x = undistorted[0, 0, 0]
    y = undistorted[0, 0, 1]

    # Ray in camera coordinates
    ray_camera = np.array(
        [x, y, 1.0],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Camera model:
    #
    # X_camera = R * X_world + t
    #
    # A point on the ray is:
    #
    # X_camera = lambda * ray_camera
    #
    # We need Z_world = 0.
    # --------------------------------------------------------

    R_inv = R.T

    camera_position_world = -R_inv @ tvec.reshape(3)

    ray_world = R_inv @ ray_camera

    # --------------------------------------------------------
    # Find intersection with Z = 0 plane
    # --------------------------------------------------------

    if abs(ray_world[2]) < 1e-10:
        return None

    scale = (
        -camera_position_world[2]
        / ray_world[2]
    )

    world_point = (
        camera_position_world
        + scale * ray_world
    )

    return world_point


# ============================================================
# MOUSE
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global selected_pixel

    if event == cv2.EVENT_LBUTTONDOWN:

        selected_pixel = (x, y)

        print()
        print("--------------------------------")
        print("Selected pixel")
        print(f"u = {x}")
        print(f"v = {y}")
        print("--------------------------------")


# ============================================================
# WINDOW
# ============================================================

window_name = "Object Coordinate"

cv2.namedWindow(window_name)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # FIND CHECKERBOARD
    # ========================================================

    found, corners = cv2.findChessboardCornersSB(
        gray,
        BOARD_SIZE
    )

    if found:

        # ----------------------------------------------------
        # Estimate checkerboard pose
        # ----------------------------------------------------

        success, rvec, tvec = cv2.solvePnP(
            object_points,
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
                frame,
                BOARD_SIZE,
                corners,
                found
            )

            # ------------------------------------------------
            # Draw origin and axes
            # ------------------------------------------------

            axis_length = 40.0

            axis_points = np.float32([
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, -axis_length],
            ])

            projected, _ = cv2.projectPoints(
                axis_points,
                rvec,
                tvec,
                camera_matrix,
                distortion
            )

            projected = projected.reshape(-1, 2)

            origin = tuple(
                projected[0].astype(int)
            )

            x_axis = tuple(
                projected[1].astype(int)
            )

            y_axis = tuple(
                projected[2].astype(int)
            )

            z_axis = tuple(
                projected[3].astype(int)
            )

            # X = red
            cv2.line(
                frame,
                origin,
                x_axis,
                (0, 0, 255),
                3
            )

            # Y = green
            cv2.line(
                frame,
                origin,
                y_axis,
                (0, 255, 0),
                3
            )

            # Z = blue
            cv2.line(
                frame,
                origin,
                z_axis,
                (255, 0, 0),
                3
            )

            # =================================================
            # OBJECT CLICK
            # =================================================

            if selected_pixel is not None:

                world_point = pixel_to_world(
                    selected_pixel,
                    rvec,
                    tvec
                )

                if world_point is not None:

                    selected_world = world_point

                    X = world_point[0]
                    Y = world_point[1]

                    # -----------------------------------------
                    # Distance from origin
                    # -----------------------------------------

                    distance = np.sqrt(
                        X * X +
                        Y * Y
                    )

                    # -----------------------------------------
                    # Draw selected pixel
                    # -----------------------------------------

                    cv2.circle(
                        frame,
                        selected_pixel,
                        8,
                        (0, 255, 255),
                        -1
                    )

                    # -----------------------------------------
                    # Draw text
                    # -----------------------------------------

                    cv2.putText(
                        frame,
                        f"X = {X:.2f} mm",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Y = {Y:.2f} mm",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Distance = {distance:.2f} mm",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    # -----------------------------------------
                    # Console
                    # -----------------------------------------

                    print(
                        f"Object: "
                        f"X={X:.2f} mm, "
                        f"Y={Y:.2f} mm, "
                        f"Distance={distance:.2f} mm"
                    )

    else:

        cv2.putText(
            frame,
            "CHECKERBOARD NOT FOUND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    h, w = frame.shape[:2]

    cv2.putText(
        frame,
        "CLICK OBJECT | ESC = exit | R = reset",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        window_name,
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ESC
    if key == 27:
        break

    # Reset
    if key == ord("r"):

        selected_pixel = None
        selected_world = None

        print("Selection reset.")


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()