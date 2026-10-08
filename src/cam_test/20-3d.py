import cv2
import numpy as np
import math
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = Path("/home/mfk01/Pictures/tests/Untitled2.png")

IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 720

MARKER_SIZE_M = 0.200

REFERENCE_ID = 53
TARGET_ID = 54

HFOV_DEG = 88.0
VFOV_DEG = 68.0


# ============================================================
# CAMERA MATRIX
# ============================================================

fx = IMAGE_WIDTH / (
    2.0 * math.tan(math.radians(HFOV_DEG) / 2.0)
)

fy = IMAGE_HEIGHT / (
    2.0 * math.tan(math.radians(VFOV_DEG) / 2.0)
)

cx = IMAGE_WIDTH / 2.0
cy = IMAGE_HEIGHT / 2.0

camera_matrix = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

dist_coeffs = np.zeros(
    (5, 1),
    dtype=np.float64
)


# ============================================================
# ARUCO
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_100
)

aruco_params = cv2.aruco.DetectorParameters()

aruco_params.cornerRefinementMethod = (
    cv2.aruco.CORNER_REFINE_SUBPIX
)

# Important for your small floor markers
aruco_params.minMarkerPerimeterRate = 0.01
aruco_params.maxMarkerPerimeterRate = 4.0
aruco_params.polygonalApproxAccuracyRate = 0.05

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    aruco_params
)


# ============================================================
# 200 mm ARUCO MARKER MODEL
# ============================================================

s = MARKER_SIZE_M / 2.0

marker_3d = np.array([
    [-s,  s, 0.0],
    [ s,  s, 0.0],
    [ s, -s, 0.0],
    [-s, -s, 0.0]
], dtype=np.float64)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise RuntimeError(
        f"Cannot load image: {IMAGE_PATH}"
    )

original = image.copy()


# ============================================================
# UPSCALE
# ============================================================

scale = 2.0

image_big = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)


# ============================================================
# GRAYSCALE + OTSU
# ============================================================

gray = cv2.cvtColor(
    image_big,
    cv2.COLOR_BGR2GRAY
)

_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# ============================================================
# ARUCO DETECTION
# ============================================================

corners, ids, rejected = detector.detectMarkers(
    binary
)


# Convert IDs to dictionary
detected = {}

if ids is not None:

    ids = ids.flatten()

    for i, marker_id in enumerate(ids):

        # Convert back from 2x image coordinates
        marker_corners = (
            corners[i].reshape(4, 2) / scale
        )

        detected[int(marker_id)] = marker_corners


# ============================================================
# PRINT DETECTION
# ============================================================

print()
print("=" * 65)
print("ARUCO DETECTION")
print("=" * 65)

if detected:

    print(
        "Detected IDs:",
        sorted(detected.keys())
    )

else:

    print("No markers detected.")


# ============================================================
# DRAW DETECTED MARKERS
# ============================================================

output = original.copy()

for marker_id, pts in detected.items():

    pts_int = pts.astype(
        np.int32
    ).reshape(
        (-1, 1, 2)
    )

    cv2.polylines(
        output,
        [pts_int],
        True,
        (0, 255, 0),
        2
    )

    center = np.mean(
        pts,
        axis=0
    )

    center_int = tuple(
        center.astype(int)
    )

    cv2.circle(
        output,
        center_int,
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        output,
        f"M{marker_id}",
        (
            center_int[0] - 20,
            center_int[1] - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# ============================================================
# CHECK M53 + M54
# ============================================================

if REFERENCE_ID not in detected:

    print()
    print("ERROR:")
    print("M53 was not detected.")

    cv2.imwrite(
        "aruco_result.png",
        output
    )

    print(
        "Saved: aruco_result.png"
    )

    raise SystemExit


if TARGET_ID not in detected:

    print()
    print("ERROR:")
    print("M54 was not detected.")

    cv2.imwrite(
        "aruco_result.png",
        output
    )

    print(
        "Saved: aruco_result.png"
    )

    raise SystemExit


# ============================================================
# POSE ESTIMATION
# ============================================================

def estimate_pose(image_points):

    success, rvec, tvec = cv2.solvePnP(
        marker_3d,
        image_points.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not success:
        raise RuntimeError(
            "solvePnP failed."
        )

    R, _ = cv2.Rodrigues(
        rvec
    )

    return R, tvec


R53, t53 = estimate_pose(
    detected[53]
)

R54, t54 = estimate_pose(
    detected[54]
)


# ============================================================
# MARKER CENTERS IN CAMERA COORDINATES
# ============================================================

# Marker center = [0, 0, 0] in marker coordinates

center53_camera = (
    t53.flatten()
)

center54_camera = (
    t54.flatten()
)


# ============================================================
# CAMERA-SPACE DIFFERENCE
# ============================================================

difference_camera = (
    center54_camera -
    center53_camera
)


# ============================================================
# EXPRESS M54 IN M53 COORDINATE SYSTEM
# ============================================================

# R53 transforms:
#
# M53 coordinates -> Camera coordinates
#
# Therefore:
#
# Camera -> M53 = R53.T

difference_m53 = (
    R53.T @ difference_camera
)


# ============================================================
# MILLIMETERS
# ============================================================

difference_mm = (
    difference_m53 * 1000.0
)


dx = difference_mm[0]
dy = difference_mm[1]
dz = difference_mm[2]

distance = np.linalg.norm(
    difference_mm
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 65)
print("M54 RELATIVE TO M53")
print("=" * 65)

print(
    f"M53 -> M54 X : {dx:10.2f} mm"
)

print(
    f"M53 -> M54 Y : {dy:10.2f} mm"
)

print(
    f"M53 -> M54 Z : {dz:10.2f} mm"
)

print(
    f"Distance      : {distance:10.2f} mm"
)

print("=" * 65)


# ============================================================
# MARKER CENTERS
# ============================================================

center53_image = np.mean(
    detected[53],
    axis=0
).astype(int)

center54_image = np.mean(
    detected[54],
    axis=0
).astype(int)


# Draw line M53 -> M54

cv2.line(
    output,
    tuple(center53_image),
    tuple(center54_image),
    (255, 0, 0),
    3
)


# ============================================================
# INFORMATION ON IMAGE
# ============================================================

lines = [
    "M53 -> M54",
    f"X = {dx:.1f} mm",
    f"Y = {dy:.1f} mm",
    f"Z = {dz:.1f} mm",
    f"Distance = {distance:.1f} mm"
]


y = 30

for line in lines:

    cv2.putText(
        output,
        line,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    y += 30


# ============================================================
# SAVE
# ============================================================

output_path = "aruco_result.png"

cv2.imwrite(
    output_path,
    output
)

print()
print(
    f"Annotated image saved to: {output_path}"
)