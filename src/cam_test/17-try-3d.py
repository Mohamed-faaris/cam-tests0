
import cv2
import numpy as np
import math

# ============================================================
# REAL-TIME 2-CAMERA CHARUCO + ARUCO 6D POSE
#
# NO CALIBRATION FILE
#
# Assumptions:
#   1. Lens distortion / warping has ALREADY been corrected.
#   2. Both cameras provide rectified/warped images.
#   3. The ChArUco board is visible in BOTH cameras.
#   4. The board itself defines WORLD coordinates.
#
# Board:
#   7 x 9 squares
#   square = 25 mm
#   marker = 18 mm
#   marker border bits = 1
#
# Runtime:
#   - Find ChArUco board in both cameras.
#   - Estimate board pose in both cameras.
#   - Derive Camera-2 relative to Camera-1 from the board.
#   - Find phone ArUco marker in both cameras.
#   - Triangulate its 4 corners.
#   - Convert 3D marker points to board/world coordinates.
#   - Estimate marker 6D pose.
#
# IMPORTANT:
#   A single planar board cannot recover arbitrary camera intrinsics
#   from one frame. Therefore this prototype assumes the warped
#   images have known focal length/principal point.
#
#   Change FX/FY/CX/CY below for your warped images.
#
# ============================================================


# ============================================================
# USER SETTINGS
# ============================================================

CAMERA_1 = 0
CAMERA_2 = 1

# Image resolution AFTER your existing warping/rectification.
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160

# ------------------------------------------------------------
# Approximate intrinsics for the ALREADY WARPED image.
#
# If your warp is based on a calibrated camera model, put the
# corresponding values here.
#
# If you don't know them yet, this gives a starting estimate
# for an ~80.9 degree horizontal FOV.
# ------------------------------------------------------------

HORIZONTAL_FOV_DEG = 80.86

FX = (
    IMAGE_WIDTH /
    (2.0 * math.tan(
        math.radians(HORIZONTAL_FOV_DEG / 2.0)
    ))
)

FY = FX

CX = IMAGE_WIDTH / 2.0
CY = IMAGE_HEIGHT / 2.0


# ------------------------------------------------------------
# ChArUco board
# ------------------------------------------------------------

BOARD_SQUARES_X = 7
BOARD_SQUARES_Y = 9

SQUARE_SIZE_M = 0.025
MARKER_SIZE_M = 0.018

ARUCO_DICTIONARY = cv2.aruco.DICT_6X6_50

# ------------------------------------------------------------
# Phone marker
# ------------------------------------------------------------

PHONE_MARKER_ID = 20

# The physical side length of the ArUco marker on the phone.
#
# CHANGE THIS.
#
# Example:
#   50 mm marker -> 0.050
# ------------------------------------------------------------

PHONE_MARKER_SIZE_M = 0.050


# Minimum board markers required for board pose.
MIN_BOARD_MARKERS = 4

# Number of frames used for temporal averaging of the
# camera relationship.
EXTRINSIC_HISTORY_LENGTH = 10

# Number of frames used for phone pose smoothing.
POSE_HISTORY_LENGTH = 8


# ============================================================
# CAMERA MATRIX
# ============================================================

K = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Distortion is assumed to already be removed.
D = np.zeros((5, 1), dtype=np.float64)


print()
print("======================================================")
print("REAL-TIME CHARUCO WORLD + STEREO ARUCO 6D POSE")
print("======================================================")
print()
print(f"Resolution: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
print(f"Estimated FX/FY: {FX:.1f}")
print(f"CX/CY: {CX:.1f}, {CY:.1f}")
print()
print("Board:")
print(f"  {BOARD_SQUARES_X} x {BOARD_SQUARES_Y} squares")
print(f"  square = {SQUARE_SIZE_M * 1000:.1f} mm")
print(f"  marker = {MARKER_SIZE_M * 1000:.1f} mm")
print()
print(f"Phone marker ID: {PHONE_MARKER_ID}")
print(
    f"Phone marker size: "
    f"{PHONE_MARKER_SIZE_M * 1000:.1f} mm"
)
print()
print("Dictionary: DICT_6X6_50")
print("Q = quit")
print("R = reset smoothing")
print()


# ============================================================
# ARUCO / CHARUCO
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(
    ARUCO_DICTIONARY
)

board = cv2.aruco.CharucoBoard(
    (BOARD_SQUARES_X, BOARD_SQUARES_Y),
    SQUARE_SIZE_M,
    MARKER_SIZE_M,
    aruco_dict
)

params = cv2.aruco.DetectorParameters()

params.markerBorderBits = 1

params.cornerRefinementMethod = (
    cv2.aruco.CORNER_REFINE_SUBPIX
)

params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 53
params.adaptiveThreshWinSizeStep = 4

params.minMarkerPerimeterRate = 0.005
params.maxMarkerPerimeterRate = 4.0

params.polygonalApproxAccuracyRate = 0.03

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    params
)


# ============================================================
# BOARD MARKER 3D GEOMETRY
# ============================================================

board_marker_object_points = board.getObjPoints()
board_marker_ids = board.getIds().flatten()

board_marker_lookup = {
    int(marker_id): np.asarray(
        board_marker_object_points[i],
        dtype=np.float64
    )
    for i, marker_id in enumerate(board_marker_ids)
}


# ============================================================
# CAMERA OPEN
# ============================================================

cap1 = cv2.VideoCapture(CAMERA_1)
cap2 = cv2.VideoCapture(CAMERA_2)

if not cap1.isOpened():
    raise RuntimeError(
        f"Cannot open Camera 1: {CAMERA_1}"
    )

if not cap2.isOpened():
    raise RuntimeError(
        f"Cannot open Camera 2: {CAMERA_2}"
    )

# Try to request the desired image size.
# Your camera/warping pipeline may override this.
for cap in (cap1, cap2):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)


# ============================================================
# DETECTION
# ============================================================

def detect_markers(frame):
    """
    The input frame is assumed to already be geometrically
    corrected/warped.

    No distortion correction is applied here.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, rejected = detector.detectMarkers(
        gray
    )

    return corners, ids, rejected


def marker_map(corners, ids):
    result = {}

    if ids is None:
        return result

    for i, marker_id in enumerate(ids.flatten()):
        result[int(marker_id)] = (
            corners[i]
            .reshape(4, 2)
            .astype(np.float64)
        )

    return result


# ============================================================
# BOARD POSE FROM ARUCO MARKERS
# ============================================================

def estimate_board_pose(markers):
    """
    Estimate:

        X_camera = R_board_to_camera X_board + t

    using the known physical geometry of the ChArUco
    board's ArUco markers.

    This intentionally does NOT call interpolateCornersCharuco().
    """

    obj = []
    img = []
    used_ids = []

    for marker_id, image_corners in markers.items():

        if marker_id not in board_marker_lookup:
            continue

        board_corners = board_marker_lookup[
            marker_id
        ]

        obj.extend(board_corners.tolist())
        img.extend(image_corners.tolist())

        used_ids.append(marker_id)

    if len(used_ids) < MIN_BOARD_MARKERS:
        return None

    obj = np.asarray(
        obj,
        dtype=np.float64
    )

    img = np.asarray(
        img,
        dtype=np.float64
    )

    # Initial solve.
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        img,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        return None

    # Refine.
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            obj,
            img,
            K,
            D,
            rvec,
            tvec
        )
    except Exception:
        pass

    R_board_camera, _ = cv2.Rodrigues(rvec)

    projected, _ = cv2.projectPoints(
        obj,
        rvec,
        tvec,
        K,
        D
    )

    projected = projected.reshape(-1, 2)

    pixel_error = np.linalg.norm(
        projected - img,
        axis=1
    )

    rms = float(
        np.sqrt(
            np.mean(pixel_error ** 2)
        )
    )

    return {
        "R": R_board_camera,
        "t": tvec,
        "rvec": rvec,
        "markers": used_ids,
        "rms": rms
    }


# ============================================================
# CAMERA RELATIONSHIP FROM BOARD
# ============================================================

def relative_camera_pose(
    board1,
    board2
):
    """
    Board pose in Camera 1:
        X1 = R1 Xw + t1

    Board pose in Camera 2:
        X2 = R2 Xw + t2

    Therefore:

        X2 = R2 R1.T X1
             + t2 - R2 R1.T t1

    This gives Camera 1 -> Camera 2.
    """

    R1 = board1["R"]
    t1 = board1["t"]

    R2 = board2["R"]
    t2 = board2["t"]

    R21 = R2 @ R1.T

    T21 = (
        t2 -
        R21 @ t1
    )

    return R21, T21


# ============================================================
# TRIANGULATION
# ============================================================

def triangulate_points(
    points1,
    points2,
    R21,
    T21
):
    """
    Input points are pixel coordinates in the already
    rectified/warped images.

    Output is in Camera 1 coordinates.
    """

    # Convert pixels -> normalized rays.
    p1 = cv2.undistortPoints(
        points1.reshape(-1, 1, 2),
        K,
        D
    ).reshape(-1, 2).T

    p2 = cv2.undistortPoints(
        points2.reshape(-1, 1, 2),
        K,
        D
    ).reshape(-1, 2).T

    P1 = np.hstack([
        np.eye(3),
        np.zeros((3, 1))
    ])

    P2 = np.hstack([
        R21,
        T21
    ])

    Xh = cv2.triangulatePoints(
        P1,
        P2,
        p1,
        p2
    )

    X = (
        Xh[:3] /
        Xh[3]
    ).T

    return X


# ============================================================
# CAMERA 1 -> WORLD / BOARD
# ============================================================

def camera_to_world(points_camera, board1):
    """
    Board/world coordinates:

        X_camera = R X_world + t

    therefore:

        X_world = R.T (X_camera - t)
    """

    R = board1["R"]
    t = board1["t"]

    points_camera = np.asarray(
        points_camera,
        dtype=np.float64
    )

    return (
        R.T @
        (
            points_camera.T -
            t
        )
    ).T


# ============================================================
# PHONE POSE
# ============================================================

def estimate_phone_pose_world(
    marker_world
):
    """
    marker_world:
        4 x 3 marker corners in board/world coordinates.

    Returns:
        center
        rotation matrix
        roll/pitch/yaw
    """

    half = PHONE_MARKER_SIZE_M / 2.0

    marker_object = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0]
    ], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        marker_object,
        marker_world,
        np.eye(3),
        None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            marker_object,
            marker_world,
            np.eye(3),
            None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)

    # The center from the triangulated corners is generally
    # more useful for this prototype than relying only on
    # solvePnP's tvec.
    center = np.mean(
        marker_world,
        axis=0
    )

    sy = math.sqrt(
        R[0, 0] ** 2 +
        R[1, 0] ** 2
    )

    if sy > 1e-6:

        roll = math.atan2(
            R[2, 1],
            R[2, 2]
        )

        pitch = math.atan2(
            -R[2, 0],
            sy
        )

        yaw = math.atan2(
            R[1, 0],
            R[0, 0]
        )

    else:

        roll = math.atan2(
            -R[1, 2],
            R[1, 1]
        )

        pitch = math.atan2(
            -R[2, 0],
            sy
        )

        yaw = 0.0

    return {
        "center": center,
        "R": R,
        "rvec": rvec,
        "roll": math.degrees(roll),
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw)
    }


# ============================================================
# SMOOTHING
# ============================================================

camera_R_history = []
camera_T_history = []

pose_position_history = []
pose_angle_history = []


def average_rotation(rotations):
    """
    Average rotations using Rodrigues vectors.
    Good enough for this prototype when frame-to-frame
    changes are small.
    """

    rvecs = []

    for R in rotations:

        rvec, _ = cv2.Rodrigues(R)
        rvecs.append(
            rvec.flatten()
        )

    mean_rvec = np.mean(
        np.asarray(rvecs),
        axis=0
    )

    Rmean, _ = cv2.Rodrigues(
        mean_rvec.reshape(3, 1)
    )

    return Rmean


def smooth_camera_pose(
    R,
    T
):
    camera_R_history.append(R.copy())
    camera_T_history.append(T.copy())

    if len(camera_R_history) > EXTRINSIC_HISTORY_LENGTH:
        camera_R_history.pop(0)
        camera_T_history.pop(0)

    Ravg = average_rotation(
        camera_R_history
    )

    Tavg = np.mean(
        np.asarray(camera_T_history),
        axis=0
    )

    return Ravg, Tavg


def smooth_phone_pose(
    position,
    angles
):
    pose_position_history.append(
        position.copy()
    )

    pose_angle_history.append(
        angles.copy()
    )

    if len(pose_position_history) > POSE_HISTORY_LENGTH:
        pose_position_history.pop(0)

    if len(pose_angle_history) > POSE_HISTORY_LENGTH:
        pose_angle_history.pop(0)

    p = np.mean(
        np.asarray(pose_position_history),
        axis=0
    )

    a = np.mean(
        np.asarray(pose_angle_history),
        axis=0
    )

    return p, a


def reset_history():
    camera_R_history.clear()
    camera_T_history.clear()

    pose_position_history.clear()
    pose_angle_history.clear()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ok1, frame1 = cap1.read()
    ok2, frame2 = cap2.read()

    if not ok1 or not ok2:
        print("Camera read error.")
        break

    # --------------------------------------------------------
    # Detect all markers.
    # --------------------------------------------------------

    corners1, ids1, rejected1 = detect_markers(
        frame1
    )

    corners2, ids2, rejected2 = detect_markers(
        frame2
    )

    markers1 = marker_map(
        corners1,
        ids1
    )

    markers2 = marker_map(
        corners2,
        ids2
    )

    # --------------------------------------------------------
    # Board pose in both cameras.
    # --------------------------------------------------------

    board1 = estimate_board_pose(
        markers1
    )

    board2 = estimate_board_pose(
        markers2
    )

    # --------------------------------------------------------
    # Display copies.
    # --------------------------------------------------------

    view1 = frame1.copy()
    view2 = frame2.copy()

    if ids1 is not None:
        cv2.aruco.drawDetectedMarkers(
            view1,
            corners1,
            ids1
        )

    if ids2 is not None:
        cv2.aruco.drawDetectedMarkers(
            view2,
            corners2,
            ids2
        )

    # --------------------------------------------------------
    # Find Camera 1 -> Camera 2 from board.
    # --------------------------------------------------------

    R21 = None
    T21 = None

    if board1 is not None and board2 is not None:

        R21_raw, T21_raw = relative_camera_pose(
            board1,
            board2
        )

        R21, T21 = smooth_camera_pose(
            R21_raw,
            T21_raw
        )

    # --------------------------------------------------------
    # Phone marker.
    # --------------------------------------------------------

    phone1 = markers1.get(
        PHONE_MARKER_ID
    )

    phone2 = markers2.get(
        PHONE_MARKER_ID
    )

    if phone1 is not None:

        cv2.polylines(
            view1,
            [
                phone1.astype(
                    np.int32
                ).reshape(-1, 1, 2)
            ],
            True,
            (0, 255, 0),
            3
        )

    if phone2 is not None:

        cv2.polylines(
            view2,
            [
                phone2.astype(
                    np.int32
                ).reshape(-1, 1, 2)
            ],
            True,
            (0, 255, 0),
            3
        )

    # --------------------------------------------------------
    # Diagnostics.
    # --------------------------------------------------------

    board_status_1 = (
        "BOARD OK"
        if board1 is not None
        else "BOARD LOST"
    )

    board_status_2 = (
        "BOARD OK"
        if board2 is not None
        else "BOARD LOST"
    )

    cv2.putText(
        view1,
        f"{board_status_1}  markers={len(markers1)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0)
        if board1 is not None
        else (0, 0, 255),
        2
    )

    cv2.putText(
        view2,
        f"{board_status_2}  markers={len(markers2)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0)
        if board2 is not None
        else (0, 0, 255),
        2
    )

    # --------------------------------------------------------
    # 3D reconstruction.
    # --------------------------------------------------------

    pose = None

    if (
        phone1 is not None and
        phone2 is not None and
        R21 is not None and
        T21 is not None and
        board1 is not None
    ):

        try:

            # Triangulate the four marker corners.
            marker_camera1 = triangulate_points(
                phone1,
                phone2,
                R21,
                T21
            )

            # Camera 1 -> board/world.
            marker_world = camera_to_world(
                marker_camera1,
                board1
            )

            # Estimate phone marker pose.
            pose = estimate_phone_pose_world(
                marker_world
            )

        except Exception as exc:

            pose = None

    # --------------------------------------------------------
    # Display 6D pose.
    # --------------------------------------------------------

    if pose is not None:

        position = pose["center"]

        angles = np.array([
            pose["roll"],
            pose["pitch"],
            pose["yaw"]
        ])

        position_s, angles_s = smooth_phone_pose(
            position,
            angles
        )

        x, y, z = (
            position_s * 1000.0
        )

        roll, pitch, yaw = angles_s

        cv2.putText(
            view1,
            "PHONE 6D POSE",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2
        )

        lines = [
            f"X: {x:8.1f} mm",
            f"Y: {y:8.1f} mm",
            f"Z: {z:8.1f} mm",
            f"Roll : {roll:7.2f} deg",
            f"Pitch: {pitch:7.2f} deg",
            f"Yaw  : {yaw:7.2f} deg"
        ]

        for i, line in enumerate(lines):

            cv2.putText(
                view1,
                line,
                (20, 112 + i * 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2
            )

        cv2.putText(
            view2,
            f"XYZ: "
            f"{x:.0f}, {y:.0f}, {z:.0f} mm",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        if board1 is None or board2 is None:

            message = (
                "BOARD MUST BE VISIBLE "
                "IN BOTH CAMERAS"
            )

        elif phone1 is None or phone2 is None:

            message = (
                "PHONE MARKER MUST BE "
                "VISIBLE IN BOTH CAMERAS"
            )

        else:

            message = (
                "WAITING FOR 3D POSE"
            )

        cv2.putText(
            view1,
            message,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # Camera baseline diagnostic.
    # --------------------------------------------------------

    if T21 is not None:

        baseline = (
            np.linalg.norm(T21) * 1000.0
        )

        cv2.putText(
            view1,
            f"Live baseline: {baseline:.1f} mm",
            (20, view1.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # --------------------------------------------------------
    # Show.
    # --------------------------------------------------------

    cv2.imshow(
        "Camera 1 - World Pose",
        view1
    )

    cv2.imshow(
        "Camera 2 - Stereo",
        view2
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("r"):
        reset_history()
        print("Reset runtime history.")


cap1.release()
cap2.release()

cv2.destroyAllWindows()

print("Stopped.")