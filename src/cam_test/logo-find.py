import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

CAMERA_ID = 0

from pathlib import Path

# Project root:
# cam-test/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"
CALIBRATION_PATH = PROJECT_ROOT / "camera_calibration.npz"

# 9 x 6 squares -> 8 x 5 internal corners
BOARD_SIZE = (8, 5)

SQUARE_SIZE_MM = 4.0

# Minimum number of good ORB matches
MIN_MATCHES = 8

# RANSAC reprojection threshold for logo homography
RANSAC_THRESHOLD = 5.0


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

calibration = np.load(CALIBRATION_PATH)

camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion"]

print("Camera matrix:")
print(camera_matrix)

print("\nDistortion:")
print(distortion)


# ============================================================
# LOAD LOGO
# ============================================================

logo = cv2.imread(
    LOGO_PATH,
    cv2.IMREAD_GRAYSCALE
)

if logo is None:
    raise RuntimeError(
        f"Could not load logo: {LOGO_PATH}"
    )

logo_h, logo_w = logo.shape

print()
print("Logo:")
print(f"Width  : {logo_w}")
print(f"Height : {logo_h}")


# ============================================================
# ORB
# ============================================================

orb = cv2.ORB_create(
    nfeatures=2000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    patchSize=31
)


# ============================================================
# FIND FEATURES IN LOGO
# ============================================================

logo_keypoints, logo_descriptors = orb.detectAndCompute(
    logo,
    None
)

if logo_descriptors is None:
    raise RuntimeError(
        "Could not find features in logo image."
    )

print(
    f"Logo features: {len(logo_keypoints)}"
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open camera"
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
# CHECKERBOARD 3D POINTS
# ============================================================

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
# MATCHER
# ============================================================

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# PIXEL -> WORLD
#
# Assumes the logo lies on Z = 0 plane.
# ============================================================

def pixel_to_world(
    pixel,
    rvec,
    tvec
):

    # --------------------------------------------------------
    # Camera rotation
    # --------------------------------------------------------

    R, _ = cv2.Rodrigues(rvec)

    # --------------------------------------------------------
    # Undistort pixel
    # --------------------------------------------------------

    point = np.array(
        [[[
            float(pixel[0]),
            float(pixel[1])
        ]]],
        dtype=np.float64
    )

    undistorted = cv2.undistortPoints(
        point,
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
    # Camera -> world
    # --------------------------------------------------------

    R_inv = R.T

    camera_position_world = (
        -R_inv @ tvec.reshape(3)
    )

    ray_world = R_inv @ ray_camera

    # --------------------------------------------------------
    # Intersect ray with Z = 0
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
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera read failed")
        break

    display = frame.copy()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
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

    if board_found:

        # ----------------------------------------------------
        # Estimate board pose
        # ----------------------------------------------------

        success, rvec, tvec = cv2.solvePnP(
            board_points,
            corners,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:

            cv2.drawChessboardCorners(
                display,
                BOARD_SIZE,
                corners,
                board_found
            )

            # ------------------------------------------------
            # Draw coordinate axes
            # ------------------------------------------------

            axis_length = 40.0

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

            # X
            cv2.line(
                display,
                origin,
                x_axis,
                (0, 0, 255),
                3
            )

            # Y
            cv2.line(
                display,
                origin,
                y_axis,
                (0, 255, 0),
                3
            )

            # Z
            cv2.line(
                display,
                origin,
                z_axis,
                (255, 0, 0),
                3
            )

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
    # 2. FIND LOGO
    # ========================================================

    logo_keypoints_scene, logo_descriptors_scene = (
        orb.detectAndCompute(
            gray,
            None
        )
    )

    logo_detected = False
    logo_center_pixel = None

    if (
        logo_descriptors_scene is not None
        and len(logo_descriptors_scene) > 0
    ):

        # ----------------------------------------------------
        # KNN matching
        # ----------------------------------------------------

        matches = matcher.knnMatch(
            logo_descriptors,
            logo_descriptors_scene,
            k=2
        )

        # ----------------------------------------------------
        # Lowe ratio test
        # ----------------------------------------------------

        good_matches = []

        for pair in matches:

            if len(pair) != 2:
                continue

            m, n = pair

            if m.distance < 0.70 * n.distance:
                good_matches.append(m)

        # ----------------------------------------------------
        # Need enough matches
        # ----------------------------------------------------

        if len(good_matches) >= MIN_MATCHES:

            src_points = np.float32([
                logo_keypoints[m.queryIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            dst_points = np.float32([
                logo_keypoints_scene[m.trainIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            # ------------------------------------------------
            # Find logo -> camera homography
            # ------------------------------------------------

            H, mask = cv2.findHomography(
                src_points,
                dst_points,
                cv2.RANSAC,
                RANSAC_THRESHOLD
            )

            if H is not None and mask is not None:

                inliers = mask.ravel().astype(bool)

                number_inliers = np.sum(inliers)

                # Require a reasonable number of inliers
                if number_inliers >= 6:

                    # ------------------------------------------------
                    # Four corners of reference logo
                    # ------------------------------------------------

                    logo_corners = np.float32([
                        [0, 0],
                        [logo_w - 1, 0],
                        [logo_w - 1, logo_h - 1],
                        [0, logo_h - 1]
                    ]).reshape(-1, 1, 2)

                    scene_corners = cv2.perspectiveTransform(
                        logo_corners,
                        H
                    )

                    # ------------------------------------------------
                    # Draw detected logo boundary
                    # ------------------------------------------------

                    scene_corners_int = np.int32(
                        scene_corners
                    )

                    cv2.polylines(
                        display,
                        [scene_corners_int],
                        True,
                        (0, 255, 0),
                        3
                    )

                    # ------------------------------------------------
                    # Find center of logo
                    # ------------------------------------------------

                    center = np.float32([
                        [
                            [logo_w / 2.0],
                            [logo_h / 2.0]
                        ]
                    ])

                    # Reshape correctly
                    center = np.array(
                        [
                            [
                                logo_w / 2.0,
                                logo_h / 2.0
                            ]
                        ],
                        dtype=np.float32
                    ).reshape(
                        -1,
                        1,
                        2
                    )

                    transformed_center = (
                        cv2.perspectiveTransform(
                            center,
                            H
                        )
                    )

                    cx, cy = (
                        transformed_center[0, 0]
                    )

                    logo_center_pixel = (
                        int(round(cx)),
                        int(round(cy))
                    )

                    logo_detected = True

                    # ------------------------------------------------
                    # Draw center
                    # ------------------------------------------------

                    cv2.circle(
                        display,
                        logo_center_pixel,
                        8,
                        (0, 255, 255),
                        -1
                    )

                    # ------------------------------------------------
                    # Display matching information
                    # ------------------------------------------------

                    cv2.putText(
                        display,
                        f"Logo matches: "
                        f"{len(good_matches)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        display,
                        f"Logo inliers: "
                        f"{number_inliers}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

    # ========================================================
    # 3. LOGO PIXEL -> BOARD COORDINATES
    # ========================================================

    if (
        logo_detected
        and rvec is not None
        and tvec is not None
    ):

        world = pixel_to_world(
            logo_center_pixel,
            rvec,
            tvec
        )

        if world is not None:

            X = world[0]
            Y = world[1]

            # ------------------------------------------------
            # Distance from origin
            # ------------------------------------------------

            distance = np.sqrt(
                X * X +
                Y * Y
            )

            # ------------------------------------------------
            # Display coordinates
            # ------------------------------------------------

            cv2.putText(
                display,
                f"X = {X:.2f} mm",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Y = {Y:.2f} mm",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Distance = {distance:.2f} mm",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            # ------------------------------------------------
            # Console output
            # ------------------------------------------------

            print(
                f"\r"
                f"Logo: "
                f"X={X:8.2f} mm   "
                f"Y={Y:8.2f} mm   "
                f"Distance={distance:8.2f} mm",
                end=""
            )

    # ========================================================
    # STATUS
    # ========================================================

    if not board_found:

        cv2.putText(
            display,
            "CHECKERBOARD NOT FOUND",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    elif not logo_detected:

        cv2.putText(
            display,
            "LOGO NOT FOUND",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            display,
            "LOGO FOUND",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Logo Distance",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()