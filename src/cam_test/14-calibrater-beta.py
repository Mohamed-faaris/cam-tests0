import cv2
import numpy as np
import math
import time

# ============================================================
# CONFIGURATION
# ============================================================

BOARD_SIZE = (8, 5)          # INNER CORNERS
SQUARE_SIZE_MM = 40.0

MIN_FRAMES = 15
MAX_FRAMES = 30

CAMERA_INDEX = 0

# Coverage grid
GRID_X = 12
GRID_Y = 8

# Quality
MIN_SHARPNESS = 80.0

# Automatic capture
MIN_CAPTURE_INTERVAL = 0.7

# A new frame must have at least this much novelty
MIN_INFORMATION_GAIN = 0.08


# ============================================================
# 3D BOARD MODEL
#
# IMPORTANT:
# The board is FLAT.
# Therefore every Z coordinate is zero.
# ============================================================

object_points_template = np.zeros(
    (BOARD_SIZE[0] * BOARD_SIZE[1], 3),
    dtype=np.float32
)

object_points_template[:, :2] = np.mgrid[
    0:BOARD_SIZE[0],
    0:BOARD_SIZE[1]
].T.reshape(-1, 2)

object_points_template *= SQUARE_SIZE_MM


# ============================================================
# DATASET
# ============================================================

selected_frames = []

# Every accepted frame's image-space corners
selected_corner_sets = []

# Feature descriptors
selected_descriptors = []

# Global corner coverage
coverage = np.zeros(
    (GRID_Y, GRID_X),
    dtype=np.float32
)

last_capture_time = 0


# ============================================================
# BASIC METRICS
# ============================================================

def sharpness_score(gray):
    """
    Laplacian variance.
    Higher means sharper.
    """
    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


def normalize(value, minimum, maximum):
    return float(
        np.clip(
            (value - minimum) /
            (maximum - minimum),
            0.0,
            1.0
        )
    )


# ============================================================
# BOARD GEOMETRY
# ============================================================

def get_board_geometry(corners, frame_shape):
    """
    Calculate basic geometry of the detected flat board.
    """

    h, w = frame_shape[:2]

    pts = corners.reshape(-1, 2)

    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])

    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    center = np.mean(pts, axis=0)

    width = max_x - min_x
    height = max_y - min_y

    area_ratio = (
        width * height
    ) / (w * h)

    center_x = center[0] / w
    center_y = center[1] / h

    return {
        "center": np.array(
            [center_x, center_y],
            dtype=np.float32
        ),
        "width": width / w,
        "height": height / h,
        "area": area_ratio
    }


# ============================================================
# BOARD ORIENTATION
# ============================================================

def board_roll(corners):
    """
    In-plane board rotation.

    Uses the first row of chessboard points.
    """

    pts = corners.reshape(
        BOARD_SIZE[1],
        BOARD_SIZE[0],
        2
    )

    p1 = pts[0, 0]
    p2 = pts[0, -1]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return math.atan2(dy, dx)


# ============================================================
# HOMOGRAPHY
# ============================================================

def calculate_homography(corners):
    """
    Because the board is planar, all board points lie on Z=0.

    Therefore image points and board coordinates are related
    by a homography.
    """

    pts = corners.reshape(-1, 2).astype(
        np.float32
    )

    planar_points = object_points_template[
        :, :2
    ].astype(np.float32)

    H, mask = cv2.findHomography(
        planar_points,
        pts,
        cv2.RANSAC,
        3.0
    )

    return H


# ============================================================
# POSE ESTIMATION
# ============================================================

def estimate_pose(corners, camera_matrix, distortion):
    """
    Estimate board pose.

    Board is planar, so the 3D model has Z=0.
    """

    if camera_matrix is None:
        return None

    success, rvec, tvec = cv2.solvePnP(
        object_points_template,
        corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)

    return {
        "rvec": rvec,
        "tvec": tvec,
        "rotation": rotation_matrix
    }


# ============================================================
# ROTATION → EULER ANGLES
# ============================================================

def rotation_to_euler(R):
    """
    Convert rotation matrix to approximate
    pitch / yaw / roll.
    """

    sy = math.sqrt(
        R[0, 0] ** 2 +
        R[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:

        x = math.atan2(
            R[2, 1],
            R[2, 2]
        )

        y = math.atan2(
            -R[2, 0],
            sy
        )

        z = math.atan2(
            R[1, 0],
            R[0, 0]
        )

    else:

        x = math.atan2(
            -R[1, 2],
            R[1, 1]
        )

        y = math.atan2(
            -R[2, 0],
            sy
        )

        z = 0

    return np.array(
        [x, y, z],
        dtype=np.float32
    )


# ============================================================
# COVERAGE MAP
# ============================================================

def points_to_grid(points, frame_shape):
    """
    Convert image coordinates to coverage-grid coordinates.
    """

    h, w = frame_shape[:2]

    grid = []

    for x, y in points:

        gx = int(
            np.clip(
                x / w * GRID_X,
                0,
                GRID_X - 1
            )
        )

        gy = int(
            np.clip(
                y / h * GRID_Y,
                0,
                GRID_Y - 1
            )
        )

        grid.append(
            (gx, gy)
        )

    return grid


def frame_coverage_gain(corners, frame_shape):
    """
    How many previously uncovered image cells
    would this frame contribute?
    """

    points = corners.reshape(-1, 2)

    cells = points_to_grid(
        points,
        frame_shape
    )

    new_cells = 0

    unique_cells = set(cells)

    for gx, gy in unique_cells:

        if coverage[gy, gx] == 0:
            new_cells += 1

    total_cells = GRID_X * GRID_Y

    return new_cells / total_cells


def update_coverage(corners, frame_shape):
    """
    Add this frame's corners to the global coverage map.
    """

    points = corners.reshape(-1, 2)

    cells = points_to_grid(
        points,
        frame_shape
    )

    for gx, gy in set(cells):

        coverage[gy, gx] = min(
            coverage[gy, gx] + 1,
            1
        )


# ============================================================
# FRAME DESCRIPTOR
# ============================================================

def create_descriptor(
    corners,
    frame_shape,
    pose=None
):
    """
    Always returns a 9-dimensional descriptor.

    [cx, cy,
     width, height,
     image_roll,
     pitch, yaw, pose_roll,
     distance]
    """

    geometry = get_board_geometry(
        corners,
        frame_shape
    )

    image_roll = board_roll(corners)

    # --------------------------------------------------------
    # Default pose values
    # --------------------------------------------------------

    pitch = 0.0
    yaw = 0.0
    pose_roll = 0.0
    distance = 0.0

    # --------------------------------------------------------
    # Real pose if available
    # --------------------------------------------------------

    if pose is not None:

        euler = rotation_to_euler(
            pose["rotation"]
        )

        pitch = euler[0] / math.pi
        yaw = euler[1] / math.pi
        pose_roll = euler[2] / math.pi

        distance_mm = np.linalg.norm(
            pose["tvec"]
        )

        # Normalize distance.
        distance = np.clip(
            distance_mm / 2000.0,
            0.0,
            5.0
        )

    # --------------------------------------------------------
    # ALWAYS 9 VALUES
    # --------------------------------------------------------

    return np.array([
        geometry["center"][0],
        geometry["center"][1],

        geometry["width"],
        geometry["height"],

        image_roll / math.pi,

        pitch,
        yaw,
        pose_roll,

        distance

    ], dtype=np.float32)
# ============================================================
# VIEWPOINT NOVELTY
# ============================================================

def viewpoint_novelty(descriptor):
    """
    Compare current viewpoint against existing frames.
    """

    if len(selected_descriptors) == 0:
        return 1.0

    distances = []

    for old in selected_descriptors:

        d = np.linalg.norm(
            descriptor - old
        )

        distances.append(d)

    # We care about the closest existing frame.
    minimum_distance = min(distances)

    return float(
        np.clip(
            minimum_distance,
            0,
            1
        )
    )


# ============================================================
# INFORMATION GAIN
# ============================================================

def information_gain(
    corners,
    frame_shape,
    descriptor
):

    coverage_gain = frame_coverage_gain(
        corners,
        frame_shape
    )

    novelty = viewpoint_novelty(
        descriptor
    )

    # Weight spatial coverage more heavily.
    score = (
        0.65 * coverage_gain +
        0.35 * novelty
    )

    return float(score)


# ============================================================
# COVERAGE QUALITY
# ============================================================

def coverage_percentage():

    return (
        np.count_nonzero(coverage) /
        coverage.size
    )


def coverage_heatmap(frame):

    h, w = frame.shape[:2]

    cell_w = w / GRID_X
    cell_h = h / GRID_Y

    for gy in range(GRID_Y):

        for gx in range(GRID_X):

            x1 = int(gx * cell_w)
            y1 = int(gy * cell_h)

            x2 = int((gx + 1) * cell_w)
            y2 = int((gy + 1) * cell_h)

            if coverage[gy, gx] > 0:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 180, 0),
                    1
                )

            else:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (80, 80, 80),
                    1
                )


# ============================================================
# FIND MISSING COVERAGE
# ============================================================

def missing_coverage_direction():

    missing = np.argwhere(
        coverage == 0
    )

    if len(missing) == 0:
        return None

    # Find the largest uncovered region.
    row, col = missing[0]

    vertical = row / GRID_Y
    horizontal = col / GRID_X

    if vertical < 0.33:
        vertical_text = "top"

    elif vertical > 0.66:
        vertical_text = "bottom"

    else:
        vertical_text = ""

    if horizontal < 0.33:
        horizontal_text = "left"

    elif horizontal > 0.66:
        horizontal_text = "right"

    else:
        horizontal_text = ""

    if horizontal_text and vertical_text:
        return (
            f"Move board toward "
            f"the {vertical_text}-{horizontal_text}"
        )

    if vertical_text:
        return (
            f"Move board toward the "
            f"{vertical_text}"
        )

    if horizontal_text:
        return (
            f"Move board toward the "
            f"{horizontal_text}"
        )

    return "Move board to a new position"


# ============================================================
# SUGGESTION ENGINE
# ============================================================

def generate_suggestion(
    corners,
    gray,
    frame_shape,
    descriptor
):

    sharpness = sharpness_score(gray)

    # --------------------------------------------------------
    # 1. Blur
    # --------------------------------------------------------

    if sharpness < MIN_SHARPNESS:

        return (
            "Hold the camera steady",
            sharpness
        )


    geometry = get_board_geometry(
        corners,
        frame_shape
    )

    cx, cy = geometry["center"]


    # --------------------------------------------------------
    # 2. Board too small
    # --------------------------------------------------------

    if geometry["area"] < 0.06:

        return (
            "Move the board closer",
            sharpness
        )


    # --------------------------------------------------------
    # 3. Board too large
    # --------------------------------------------------------

    if geometry["area"] > 0.65:

        return (
            "Move the board farther away",
            sharpness
        )


    # --------------------------------------------------------
    # 4. Edge coverage
    # --------------------------------------------------------

    if cx < 0.15:

        return (
            "Move the board slightly right",
            sharpness
        )

    if cx > 0.85:

        return (
            "Move the board slightly left",
            sharpness
        )

    if cy < 0.15:

        return (
            "Move the board slightly down",
            sharpness
        )

    if cy > 0.85:

        return (
            "Move the board slightly up",
            sharpness
        )


    # --------------------------------------------------------
    # 5. Perspective diversity
    # --------------------------------------------------------

    novelty = viewpoint_novelty(
        descriptor
    )

    if novelty < 0.04:

        return (
            "Try a different board angle",
            sharpness
        )


    # --------------------------------------------------------
    # 6. Coverage
    # --------------------------------------------------------

    missing = missing_coverage_direction()

    if missing is not None:

        return (
            missing,
            sharpness
        )


    # --------------------------------------------------------
    # 7. General perspective suggestion
    # --------------------------------------------------------

    if len(selected_frames) < 8:

        return (
            "Tilt the board to create perspective",
            sharpness
        )


    # --------------------------------------------------------
    # 8. Good
    # --------------------------------------------------------

    return (
        "Good viewpoint",
        sharpness
    )


# ============================================================
# DRAW COVERAGE
# ============================================================

def draw_ui(
    frame,
    corners,
    suggestion,
    sharpness,
    info_gain
):

    h, w = frame.shape[:2]

    # Coverage grid
    coverage_heatmap(frame)

    # Chessboard
    if corners is not None:

        cv2.drawChessboardCorners(
            frame,
            BOARD_SIZE,
            corners,
            True
        )

    # Header
    cv2.rectangle(
        frame,
        (0, 0),
        (w, 150),
        (0, 0, 0),
        -1
    )

    coverage_value = (
        coverage_percentage() * 100
    )

    lines = [
        f"Frames: {len(selected_frames)}/{MAX_FRAMES}",
        f"Coverage: {coverage_value:.1f}%",
        f"Sharpness: {sharpness:.0f}",
        f"Information gain: {info_gain:.2f}",
        suggestion
    ]

    for i, text in enumerate(lines):

        cv2.putText(
            frame,
            text,
            (20, 30 + i * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


# ============================================================
# CALIBRATION
# ============================================================

def calibrate(
    object_points,
    image_points,
    image_size
):

    print()
    print("=" * 60)
    print("RUNNING CAMERA CALIBRATION")
    print("=" * 60)

    rms, camera_matrix, distortion, rvecs, tvecs = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )
    )

    print()
    print("RMS error:")
    print(rms)

    print()
    print("Camera matrix:")
    print(camera_matrix)

    print()
    print("Distortion:")
    print(distortion)

    return {
        "rms": rms,
        "camera_matrix": camera_matrix,
        "distortion": distortion,
        "rvecs": rvecs,
        "tvecs": tvecs
    }


# ============================================================
# MAIN
# ============================================================

def main():

    global last_capture_time

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return


    object_points = []
    image_points = []

    # --------------------------------------------------------
    # We don't know the camera matrix initially.
    # Therefore pose estimation cannot be used immediately.
    # --------------------------------------------------------

    camera_matrix = None
    distortion = None

    calibration_result = None


    print()
    print("Real-time camera calibration")
    print()
    print("Board:")
    print(
        f"{BOARD_SIZE[0]} x "
        f"{BOARD_SIZE[1]} inner corners"
    )

    print(
        f"Square size: "
        f"{SQUARE_SIZE_MM} mm"
    )

    print()
    print("Q = quit")
    print("C = recalibrate")
    print()


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ====================================================
        # FIND CHESSBOARD
        # ====================================================

        found, corners = cv2.findChessboardCorners(
            gray,
            BOARD_SIZE,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                |
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        )


        suggestion = (
            "Show the calibration board"
        )

        sharpness = sharpness_score(
            gray
        )

        info_gain = 0.0


        if found:

            # ------------------------------------------------
            # Refine corners
            # ------------------------------------------------

            criteria = (
                cv2.TERM_CRITERIA_EPS
                |
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )


            # ------------------------------------------------
            # Pose
            #
            # Only possible after we have an approximate
            # camera calibration.
            # ------------------------------------------------

            pose = None

            if camera_matrix is not None:

                pose = estimate_pose(
                    corners,
                    camera_matrix,
                    distortion
                )


            # ------------------------------------------------
            # Descriptor
            # ------------------------------------------------

            descriptor = create_descriptor(
                corners,
                frame.shape,
                pose
            )


            # ------------------------------------------------
            # Information gain
            # ------------------------------------------------

            info_gain = information_gain(
                corners,
                frame.shape,
                descriptor
            )


            # ------------------------------------------------
            # Suggestion
            # ------------------------------------------------

            suggestion, sharpness = (
                generate_suggestion(
                    corners,
                    gray,
                    frame.shape,
                    descriptor
                )
            )


            # =================================================
            # AUTOMATIC FRAME SELECTION
            # =================================================

            now = time.time()

            enough_time = (
                now - last_capture_time
                >= MIN_CAPTURE_INTERVAL
            )

            good_sharpness = (
                sharpness >= MIN_SHARPNESS
            )

            useful_frame = (
                info_gain >= MIN_INFORMATION_GAIN
            )


            if (
                good_sharpness
                and useful_frame
                and enough_time
                and len(selected_frames) < MAX_FRAMES
            ):

                selected_frames.append(
                    frame.copy()
                )

                selected_corner_sets.append(
                    corners.copy()
                )

                selected_descriptors.append(
                    descriptor.copy()
                )

                object_points.append(
                    object_points_template.copy()
                )

                image_points.append(
                    corners.copy()
                )

                update_coverage(
                    corners,
                    frame.shape
                )

                last_capture_time = now

                suggestion = (
                    "FRAME ACCEPTED"
                )

                print(
                    f"[+] Frame "
                    f"{len(selected_frames)} "
                    f"| gain={info_gain:.3f} "
                    f"| sharpness={sharpness:.0f}"
                )


        # ====================================================
        # RUN CALIBRATION
        # ====================================================

        if (
            len(image_points) >= MIN_FRAMES
            and calibration_result is None
        ):

            calibration_result = calibrate(
                object_points,
                image_points,
                (
                    frame.shape[1],
                    frame.shape[0]
                )
            )

            camera_matrix = (
                calibration_result[
                    "camera_matrix"
                ]
            )

            distortion = (
                calibration_result[
                    "distortion"
                ]
            )


            # Now that we have a camera matrix,
            # future frames can use solvePnP.


        # ====================================================
        # DATASET COMPLETE
        # ====================================================

        if len(selected_frames) >= MAX_FRAMES:

            suggestion = (
                "Dataset complete — press C to recalibrate"
            )


        # ====================================================
        # DRAW
        # ====================================================

        draw_ui(
            frame,
            corners if found else None,
            suggestion,
            sharpness,
            info_gain
        )


        cv2.imshow(
            "Intelligent Camera Calibration",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):
            break


        if key == ord("c"):

            if len(image_points) >= MIN_FRAMES:

                calibration_result = calibrate(
                    object_points,
                    image_points,
                    (
                        frame.shape[1],
                        frame.shape[0]
                    )
                )

                camera_matrix = (
                    calibration_result[
                        "camera_matrix"
                    ]
                )

                distortion = (
                    calibration_result[
                        "distortion"
                    ]
                )


    cap.release()
    cv2.destroyAllWindows()


    # ========================================================
    # SAVE
    # ========================================================

    if calibration_result is not None:

        np.savez(
            "camera_calibration.npz",

            camera_matrix=(
                calibration_result[
                    "camera_matrix"
                ]
            ),

            distortion=(
                calibration_result[
                    "distortion"
                ]
            ),

            rms=(
                calibration_result[
                    "rms"
                ]
            )
        )

        print()
        print(
            "Saved camera_calibration.npz"
        )


if __name__ == "__main__":
    main()