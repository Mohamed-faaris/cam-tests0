import cv2
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# logo-find.py
#     ↓ parents[0] = cam_test
#     ↓ parents[1] = src
#     ↓ parents[2] = cam-test project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"
CALIBRATION_PATH = PROJECT_ROOT / "camera_calibration.npz"


# ============================================================
# CONFIG
# ============================================================

CAMERA_ID = 0

# 9 x 6 checkerboard squares
# therefore 8 x 5 internal corners
BOARD_SIZE = (8, 5)

# Your checkerboard square size
SQUARE_SIZE_MM = 40.0

# ORB
MIN_MATCHES = 4

# RANSAC
RANSAC_THRESHOLD = 2.0


# ============================================================
# CHECK FILES
# ============================================================

if not LOGO_PATH.exists():
    raise FileNotFoundError(
        f"Logo not found:\n{LOGO_PATH}"
    )

if not CALIBRATION_PATH.exists():
    raise FileNotFoundError(
        f"Calibration not found:\n{CALIBRATION_PATH}"
    )


print("=" * 60)
print("LOGO POSITION TEST")
print("=" * 60)

print(f"Project root: {PROJECT_ROOT}")
print(f"Logo:         {LOGO_PATH}")
print(f"Calibration:  {CALIBRATION_PATH}")


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

calibration = np.load(
    str(CALIBRATION_PATH)
)

camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion"]

print("\nCamera matrix:")
print(camera_matrix)

print("\nDistortion:")
print(distortion)


# ============================================================
# LOAD LOGO
# ============================================================

logo = cv2.imread(
    str(LOGO_PATH),
    cv2.IMREAD_GRAYSCALE
)

if logo is None:
    raise RuntimeError(
        f"Could not load logo:\n{LOGO_PATH}"
    )

logo_height, logo_width = logo.shape

print("\nLogo:")
print(f"Width  : {logo_width}px")
print(f"Height : {logo_height}px")


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
# FIND FEATURES IN REFERENCE LOGO
# ============================================================

logo_keypoints, logo_descriptors = (
    orb.detectAndCompute(
        logo,
        None
    )
)

if logo_descriptors is None:
    raise RuntimeError(
        "Could not find ORB features in logo."
    )

print(
    f"\nLogo features: {len(logo_keypoints)}"
)


# ============================================================
# CHECKERBOARD 3D COORDINATES
#
# Origin:
#
#             Y
#             ↑
#             |
#             |
#             O ─────────→ X
#
# Z = 0 is the checkerboard plane
#
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
# MATCHER
# ============================================================

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# PIXEL -> WORLD COORDINATE
#
# Input:
#     pixel = (u, v)
#
# Output:
#     X, Y, Z in checkerboard coordinate system
#
# Assumes pixel belongs to the Z = 0 plane.
# ============================================================

def pixel_to_world(
    pixel,
    rvec,
    tvec
):

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
        [[[
            float(pixel[0]),
            float(pixel[1])
        ]]],
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

    # Ray in camera coordinates
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
        print("\nCamera read failed.")
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
            # IMPORTANT:
            # Convert numpy floats to Python ints.
            # This fixes cv2.line() error.
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
    # 2. FIND LOGO FEATURES IN CAMERA IMAGE
    # ========================================================

    scene_keypoints, scene_descriptors = (
        orb.detectAndCompute(
            gray,
            None
        )
    )

    logo_detected = False
    logo_center_pixel = None

    good_matches = []
    number_inliers = 0

    if (
        scene_descriptors is not None
        and len(scene_descriptors) > 0
    ):

        # ----------------------------------------------------
        # KNN matching
        # ----------------------------------------------------

        matches = matcher.knnMatch(
            logo_descriptors,
            scene_descriptors,
            k=2
        )

        # ----------------------------------------------------
        # Lowe ratio test
        # ----------------------------------------------------

        for pair in matches:

            if len(pair) != 2:
                continue

            m, n = pair

            if m.distance < 0.70 * n.distance:
                good_matches.append(m)

        # ----------------------------------------------------
        # Homography
        # ----------------------------------------------------

        if len(good_matches) >= MIN_MATCHES:

            src_points = np.float32(
                [
                    logo_keypoints[
                        m.queryIdx
                    ].pt
                    for m in good_matches
                ]
            ).reshape(
                -1,
                1,
                2
            )

            dst_points = np.float32(
                [
                    scene_keypoints[
                        m.trainIdx
                    ].pt
                    for m in good_matches
                ]
            ).reshape(
                -1,
                1,
                2
            )

            H, mask = cv2.findHomography(
                src_points,
                dst_points,
                cv2.RANSAC,
                RANSAC_THRESHOLD
            )

            if (
                H is not None
                and mask is not None
            ):

                inlier_mask = (
                    mask.ravel().astype(bool)
                )

                number_inliers = int(
                    np.sum(inlier_mask)
                )

                # ------------------------------------------------
                # Require enough geometrically consistent matches
                # ------------------------------------------------

                if number_inliers >= 6:

                    # ============================================
                    # LOGO CORNERS
                    # ============================================

                    logo_corners = np.float32(
                        [
                            [0, 0],
                            [logo_width - 1, 0],
                            [logo_width - 1, logo_height - 1],
                            [0, logo_height - 1]
                        ]
                    ).reshape(
                        -1,
                        1,
                        2
                    )

                    scene_corners = (
                        cv2.perspectiveTransform(
                            logo_corners,
                            H
                        )
                    )

                    scene_corners_int = (
                        np.round(
                            scene_corners
                        ).astype(
                            np.int32
                        )
                    )

                    # ------------------------------------------------
                    # Draw logo boundary
                    # ------------------------------------------------

                    cv2.polylines(
                        display,
                        [
                            scene_corners_int
                        ],
                        True,
                        (0, 255, 0),
                        3
                    )

                    # ============================================
                    # LOGO CENTER
                    # ============================================

                    logo_center = np.array(
                        [
                            [
                                [logo_width / 2.0,
                                 logo_height / 2.0]
                            ]
                        ],
                        dtype=np.float32
                    )

                    transformed_center = (
                        cv2.perspectiveTransform(
                            logo_center,
                            H
                        )
                    )

                    cx = float(
                        transformed_center[0, 0, 0]
                    )

                    cy = float(
                        transformed_center[0, 0, 1]
                    )

                    logo_center_pixel = (
                        int(round(cx)),
                        int(round(cy))
                    )

                    logo_detected = True

                    # ------------------------------------------------
                    # Draw logo center
                    # ------------------------------------------------

                    cv2.circle(
                        display,
                        logo_center_pixel,
                        8,
                        (0, 255, 255),
                        -1
                    )

    # ========================================================
    # 3. CONVERT LOGO CENTER TO BOARD COORDINATES
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
            # DISPLAY
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
                f"Z = {Z:.2f} mm",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Distance = {distance:.2f} mm",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            # ------------------------------------------------
            # Console
            # ------------------------------------------------

            print(
                f"\r"
                f"LOGO FOUND | "
                f"X={X:8.2f} mm | "
                f"Y={Y:8.2f} mm | "
                f"Z={Z:8.2f} mm | "
                f"D={distance:8.2f} mm | "
                f"matches={len(good_matches)} | "
                f"inliers={number_inliers}",
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
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    elif not logo_detected:

        cv2.putText(
            display,
            "LOGO NOT FOUND",
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            display,
            "LOGO FOUND",
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

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

print("\n")
print("=" * 60)
print("EXIT")
print("=" * 60)