import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BOARD_SIZE = (8, 5)          # Internal checkerboard corners
SQUARE_SIZE_MM = 40.0        # Size of one square in mm

# Size of bounding box created around the clicked point
TRACK_BOX_SIZE = 80


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
#             Y
#             ^
#             |
#             |
#             O ------------> X
#
# Z points out of the checkerboard plane
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

tracker = None
tracking = False

bbox = None


# ============================================================
# CREATE TRACKER
# ============================================================

def create_tracker():

    # Different OpenCV versions expose trackers differently
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()

    else:
        raise RuntimeError(
            "CSRT tracker not available. "
            "Install opencv-contrib-python."
        )


# ============================================================
# PIXEL -> WORLD COORDINATE
#
# Object is assumed to be on Z = 0 plane.
# ============================================================

def pixel_to_world(pixel, rvec, tvec):

    # --------------------------------------------------------
    # Rotation matrix
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
    # --------------------------------------------------------

    R_inv = R.T

    camera_position_world = -R_inv @ tvec.reshape(3)

    ray_world = R_inv @ ray_camera

    # --------------------------------------------------------
    # Intersect ray with Z = 0 plane
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
# WORLD -> PIXEL
#
# Used to draw world-coordinate lines on the image.
# ============================================================

def world_to_pixel(world_point, rvec, tvec):

    point_3d = np.array(
        [world_point],
        dtype=np.float32
    )

    projected, _ = cv2.projectPoints(
        point_3d,
        rvec,
        tvec,
        camera_matrix,
        distortion
    )

    return tuple(
        projected[0, 0].astype(int)
    )


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global selected_pixel
    global tracker
    global tracking
    global bbox

    if event == cv2.EVENT_LBUTTONDOWN:

        selected_pixel = (x, y)

        print()
        print("--------------------------------")
        print("Object selected")
        print(f"Pixel: u={x}, v={y}")
        print("--------------------------------")

        # ----------------------------------------------------
        # Create bounding box around clicked point
        # ----------------------------------------------------

        half = TRACK_BOX_SIZE // 2

        x1 = max(0, x - half)
        y1 = max(0, y - half)

        bbox = (
            x1,
            y1,
            TRACK_BOX_SIZE,
            TRACK_BOX_SIZE
        )

        # ----------------------------------------------------
        # Create and initialize tracker
        # ----------------------------------------------------

        tracker = create_tracker()

        # We need the current frame
        frame = param["frame"]

        success = tracker.init(
            frame,
            bbox
        )

        if success is None:
            # Some OpenCV versions return None on success
            tracking = True
        else:
            tracking = success

        print("Tracking started.")


# ============================================================
# WINDOW
# ============================================================

window_name = "Object Tracking + Coordinate System"

cv2.namedWindow(window_name)

# The frame is passed through param
callback_param = {
    "frame": None
}

cv2.setMouseCallback(
    window_name,
    mouse_callback,
    callback_param
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Give current frame to mouse callback
    callback_param["frame"] = frame.copy()

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

        # ====================================================
        # ESTIMATE CHECKERBOARD POSE
        # ====================================================

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            corners,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )


        if success:

            # =================================================
            # DRAW CHECKERBOARD
            # =================================================

            cv2.drawChessboardCorners(
                frame,
                BOARD_SIZE,
                corners,
                found
            )


            # =================================================
            # DRAW X/Y/Z AXES
            # =================================================

            axis_length = 80.0

            axis_points = np.float32([
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, -axis_length]
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


            # -------------------------------------------------
            # X AXIS
            # -------------------------------------------------

            cv2.line(
                frame,
                origin,
                x_axis,
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                "X",
                x_axis,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


            # -------------------------------------------------
            # Y AXIS
            # -------------------------------------------------

            cv2.line(
                frame,
                origin,
                y_axis,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                "Y",
                y_axis,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            # -------------------------------------------------
            # Z AXIS
            # -------------------------------------------------

            cv2.line(
                frame,
                origin,
                z_axis,
                (255, 0, 0),
                3
            )

            cv2.putText(
                frame,
                "Z",
                z_axis,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )


            # =================================================
            # OBJECT TRACKING
            # =================================================

            if tracking and tracker is not None:

                track_success, new_bbox = tracker.update(frame)

                if track_success:

                    bbox = new_bbox

                    # -----------------------------------------
                    # Bounding box
                    # -----------------------------------------

                    bx, by, bw, bh = [
                        int(v) for v in bbox
                    ]

                    cv2.rectangle(
                        frame,
                        (bx, by),
                        (bx + bw, by + bh),
                        (0, 255, 255),
                        3
                    )


                    # -----------------------------------------
                    # Object center
                    # -----------------------------------------

                    center_x = int(
                        bx + bw / 2
                    )

                    center_y = int(
                        by + bh / 2
                    )

                    selected_pixel = (
                        center_x,
                        center_y
                    )


                    # Draw center
                    cv2.circle(
                        frame,
                        selected_pixel,
                        7,
                        (0, 255, 255),
                        -1
                    )


                    # =================================================
                    # PIXEL -> WORLD
                    # =================================================

                    world_point = pixel_to_world(
                        selected_pixel,
                        rvec,
                        tvec
                    )


                    if world_point is not None:

                        selected_world = world_point

                        X = float(world_point[0])
                        Y = float(world_point[1])


                        # =================================================
                        # DISTANCE FROM ORIGIN
                        # =================================================

                        distance = np.sqrt(
                            X * X +
                            Y * Y
                        )


                        # =================================================
                        # WORLD POINT ON X AXIS
                        #
                        # (X, 0, 0)
                        # =================================================

                        x_projection_world = np.array([
                            X,
                            0,
                            0
                        ])


                        # =================================================
                        # WORLD POINT ON Y AXIS
                        #
                        # (0, Y, 0)
                        # =================================================

                        y_projection_world = np.array([
                            0,
                            Y,
                            0
                        ])


                        # Convert these points back to image pixels
                        object_pixel = world_to_pixel(
                            np.array([X, Y, 0]),
                            rvec,
                            tvec
                        )

                        x_projection_pixel = world_to_pixel(
                            x_projection_world,
                            rvec,
                            tvec
                        )

                        y_projection_pixel = world_to_pixel(
                            y_projection_world,
                            rvec,
                            tvec
                        )


                        # =================================================
                        # DRAW DISTANCE TO X AXIS
                        #
                        # Object -> X-axis projection
                        #
                        # This represents |Y|
                        # =================================================

                        cv2.line(
                            frame,
                            object_pixel,
                            x_projection_pixel,
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Y = {abs(Y):.1f} mm",
                            x_projection_pixel,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2
                        )


                        # =================================================
                        # DRAW DISTANCE TO Y AXIS
                        #
                        # Object -> Y-axis projection
                        #
                        # This represents |X|
                        # =================================================

                        cv2.line(
                            frame,
                            object_pixel,
                            y_projection_pixel,
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"X = {abs(X):.1f} mm",
                            y_projection_pixel,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2
                        )


                        # =================================================
                        # LINE FROM ORIGIN TO OBJECT
                        # =================================================

                        cv2.line(
                            frame,
                            origin,
                            object_pixel,
                            (255, 255, 0),
                            2
                        )


                        # =================================================
                        # DISPLAY COORDINATES
                        # =================================================

                        cv2.putText(
                            frame,
                            f"X = {X:.2f} mm",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Y = {Y:.2f} mm",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
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

                        cv2.putText(
                            frame,
                            "TRACKING",
                            (20, 145),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )


                else:

                    # =============================================
                    # TRACKING LOST
                    # =============================================

                    tracking = False

                    cv2.putText(
                        frame,
                        "TRACKING LOST",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
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


    # ============================================================
    # INSTRUCTIONS
    # ============================================================

    h, w = frame.shape[:2]

    cv2.putText(
        frame,
        "CLICK OBJECT | R = reset | ESC = exit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ============================================================
    # DISPLAY
    # ============================================================

    cv2.imshow(
        window_name,
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # ============================================================
    # ESC
    # ============================================================

    if key == 27:
        break


    # ============================================================
    # RESET
    # ============================================================

    if key == ord("r"):

        selected_pixel = None
        selected_world = None

        tracker = None
        tracking = False
        bbox = None

        print("Tracking reset.")


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()