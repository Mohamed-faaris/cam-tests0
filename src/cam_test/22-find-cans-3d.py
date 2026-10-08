import cv2
import numpy as np
import math
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = Path("/home/mfk01/Pictures/tests/Untitled1.png")

OUTPUT_IMAGE = Path(
    "cans_3d_detected.png"
)

# Image
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 720

# Camera FOV
HFOV_DEG = 88.0
VFOV_DEG = 68.0

# ArUco marker
MARKER_SIZE_M = 0.200

REFERENCE_MARKER = 53
SECOND_MARKER = 54

# Can dimensions
CAN_DIAMETER_M = 0.066
CAN_HEIGHT_M = 0.122

# Shelf spacing
SHELF_SPACING_M = 0.300

# ------------------------------------------------------------
# IMPORTANT
#
# Height of the FIRST shelf surface relative to M53.
#
# M53 is on the floor, so Z=0 at the center of M53.
#
# CHANGE THIS VALUE to your actual simulator geometry.
#
# Example:
# If bottom shelf surface is 600 mm above floor:
#
#     FIRST_SHELF_Z_M = 0.600
# ------------------------------------------------------------

FIRST_SHELF_Z_M = 0.600


# ============================================================
# CAMERA MATRIX
# ============================================================

fx = IMAGE_WIDTH / (
    2.0 * math.tan(
        math.radians(HFOV_DEG) / 2.0
    )
)

fy = IMAGE_HEIGHT / (
    2.0 * math.tan(
        math.radians(VFOV_DEG) / 2.0
    )
)

cx = IMAGE_WIDTH / 2.0
cy = IMAGE_HEIGHT / 2.0

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST = np.zeros(
    (5, 1),
    dtype=np.float64
)


print()
print("Camera matrix:")
print(K)


# ============================================================
# ARUCO SETUP
# ============================================================

aruco_dictionary = (
    cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_100
    )
)

aruco_parameters = (
    cv2.aruco.DetectorParameters()
)

aruco_parameters.cornerRefinementMethod = (
    cv2.aruco.CORNER_REFINE_SUBPIX
)

aruco_parameters.minMarkerPerimeterRate = 0.01
aruco_parameters.maxMarkerPerimeterRate = 4.0
aruco_parameters.polygonalApproxAccuracyRate = 0.05

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters
)


# ============================================================
# ARUCO 3D MODEL
# ============================================================

half = MARKER_SIZE_M / 2.0

marker_points = np.array([
    [-half,  half, 0.0],
    [ half,  half, 0.0],
    [ half, -half, 0.0],
    [-half, -half, 0.0]
], dtype=np.float64)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise RuntimeError(
        f"Cannot load {IMAGE_PATH}"
    )

output = image.copy()


# ============================================================
# ARUCO DETECTION
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# Upscale because the floor markers are relatively small
scale = 2.0

gray_big = cv2.resize(
    gray,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)

# Otsu improves the low-contrast floor markers
_, binary = cv2.threshold(
    gray_big,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

corners_big, ids, rejected = (
    aruco_detector.detectMarkers(
        binary
    )
)

markers = {}

if ids is not None:

    ids = ids.flatten()

    for i, marker_id in enumerate(ids):

        # Back to original image coordinates
        pts = (
            corners_big[i].reshape(4, 2)
            / scale
        )

        markers[int(marker_id)] = pts


print()
print("=" * 70)
print("ARUCO DETECTION")
print("=" * 70)

print(
    "Detected markers:",
    sorted(markers.keys())
)


# ============================================================
# DRAW ARUCO MARKERS
# ============================================================

for marker_id, pts in markers.items():

    pts_int = pts.astype(
        np.int32
    ).reshape((-1, 1, 2))

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

    center = center.astype(int)

    cv2.circle(
        output,
        tuple(center),
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        output,
        f"M{marker_id}",
        (
            center[0] - 20,
            center[1] - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# ============================================================
# REQUIRE M53
# ============================================================

if REFERENCE_MARKER not in markers:

    cv2.imwrite(
        str(OUTPUT_IMAGE),
        output
    )

    raise RuntimeError(
        "M53 was not detected."
    )


# ============================================================
# MARKER POSE
# ============================================================

def estimate_marker_pose(
    image_points
):

    success, rvec, tvec = cv2.solvePnP(
        marker_points,
        image_points.astype(
            np.float64
        ),
        K,
        DIST,
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


R53, t53 = estimate_marker_pose(
    markers[REFERENCE_MARKER]
)


# ============================================================
# OPTIONAL M54 POSE
# ============================================================

if SECOND_MARKER in markers:

    R54, t54 = estimate_marker_pose(
        markers[SECOND_MARKER]
    )

    print()
    print(
        "M54 detected and pose estimated."
    )

else:

    R54 = None
    t54 = None

    print(
        "M54 not detected."
    )


# ============================================================
# CAMERA POSITION IN M53 COORDINATES
# ============================================================

# M53 -> camera:
#
# X_camera = R53 * X_M53 + t53
#
# Camera center:
#
# C = -R^T * t

camera_position_m53 = (
    -R53.T @ t53
).flatten()


print()
print("=" * 70)
print("CAMERA POSITION RELATIVE TO M53")
print("=" * 70)

print(
    f"X = {camera_position_m53[0] * 1000:.2f} mm"
)

print(
    f"Y = {camera_position_m53[1] * 1000:.2f} mm"
)

print(
    f"Z = {camera_position_m53[2] * 1000:.2f} mm"
)


# ============================================================
# PIXEL -> RAY IN CAMERA COORDINATES
# ============================================================

K_INV = np.linalg.inv(K)


def pixel_to_camera_ray(
    u,
    v
):

    pixel = np.array([
        [u],
        [v],
        [1.0]
    ])

    ray = K_INV @ pixel

    ray = ray.flatten()

    ray /= np.linalg.norm(
        ray
    )

    return ray


# ============================================================
# CAMERA RAY -> M53 RAY
# ============================================================

def pixel_to_world_ray(
    u,
    v
):

    ray_camera = pixel_to_camera_ray(
        u,
        v
    )

    # Camera -> M53
    ray_m53 = (
        R53.T @ ray_camera
    )

    return (
        camera_position_m53,
        ray_m53
    )


# ============================================================
# INTERSECT RAY WITH Z PLANE
# ============================================================

def ray_intersect_z_plane(
    origin,
    direction,
    z
):

    dz = direction[2]

    if abs(dz) < 1e-9:
        return None

    distance = (
        z - origin[2]
    ) / dz

    if distance <= 0:
        return None

    point = (
        origin +
        distance * direction
    )

    return point


# ============================================================
# SHELF LEVEL
# ============================================================

def shelf_surface_z(
    shelf
):

    return (
        FIRST_SHELF_Z_M +
        (shelf - 1) *
        SHELF_SPACING_M
    )


# ============================================================
# CAN CENTER HEIGHT
# ============================================================
#
# We want the CENTER of the can, not the shelf surface.
#
# Therefore:
#
# can center Z =
#
# shelf surface Z + can height / 2
#
# ============================================================

def can_center_z(
    shelf
):

    return (
        shelf_surface_z(shelf)
        +
        CAN_HEIGHT_M / 2.0
    )


# ============================================================
# CAN DETECTION
# ============================================================

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

H, S, V = cv2.split(hsv)


# Colored cans
colored_mask = (
    (S > 20) &
    (V > 100)
).astype(np.uint8) * 255


# Black cans
black_mask = (
    (V < 130) &
    (S < 100)
).astype(np.uint8) * 255


# Only shelf region
colored_mask[:245] = 0
colored_mask[455:] = 0

black_mask[:245] = 0
black_mask[455:] = 0


# ============================================================
# COMPONENT DETECTOR
# ============================================================

def find_components(
    mask,
    detection_type
):

    components = []

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    for i in range(
        1,
        num_labels
    ):

        x = int(
            stats[
                i,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                i,
                cv2.CC_STAT_TOP
            ]
        )

        w = int(
            stats[
                i,
                cv2.CC_STAT_WIDTH
            ]
        )

        h = int(
            stats[
                i,
                cv2.CC_STAT_HEIGHT
            ]
        )

        area = int(
            stats[
                i,
                cv2.CC_STAT_AREA
            ]
        )

        center_x = float(
            centroids[i][0]
        )

        center_y = float(
            centroids[i][1]
        )

        if area < 80:
            continue

        if area > 600:
            continue

        if w < 8 or w > 25:
            continue

        if h < 18 or h > 35:
            continue

        components.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "center_x": center_x,
            "center_y": center_y,
            "type": detection_type
        })

    return components


cans = (
    find_components(
        colored_mask,
        "colored"
    )
    +
    find_components(
        black_mask,
        "black"
    )
)


# ============================================================
# COLOR CLASSIFICATION
# ============================================================

def classify_color(can):

    if can["type"] == "black":
        return "black"

    x = can["x"]
    y = can["y"]
    w = can["w"]
    h = can["h"]

    roi = hsv[
        y:y+h,
        x:x+w
    ]

    roi_H, roi_S, roi_V = (
        cv2.split(roi)
    )

    valid = (
        (roi_S > 20) &
        (roi_V > 100)
    )

    hue_values = roi_H[valid]

    if len(hue_values) == 0:
        return "unknown"

    hue = float(
        np.median(
            hue_values
        )
    )

    if hue < 10 or hue >= 170:
        return "red"

    if hue < 25:
        return "orange"

    if hue < 90:
        return "green"

    if hue < 140:
        return "blue"

    return "unknown"


# ============================================================
# SHELF DETECTION
# ============================================================

SHELF_ROWS = [
    (245, 300),
    (305, 355),
    (360, 405),
    (410, 455)
]


def get_shelf_level(
    y
):

    for level, (
        y_min,
        y_max
    ) in enumerate(
        SHELF_ROWS,
        start=1
    ):

        if (
            y_min <= y <= y_max
        ):
            return level

    return None


# ============================================================
# ASSIGN CAN INFORMATION
# ============================================================

for can in cans:

    can["color"] = classify_color(
        can
    )

    can["shelf"] = get_shelf_level(
        can["center_y"]
    )


# Remove invalid
cans = [
    c for c in cans
    if c["shelf"] is not None
    and c["color"] != "unknown"
]


# Sort
cans.sort(
    key=lambda c: (
        c["shelf"],
        c["center_x"]
    )
)


# Assign IDs
for i, can in enumerate(
    cans,
    start=1
):

    can["id"] = i


# ============================================================
# CALCULATE 3D CAN POSITION
# ============================================================

for can in cans:

    u = can["center_x"]
    v = can["center_y"]

    # --------------------------------------------------------
    # Ray from camera through can
    # --------------------------------------------------------

    ray_origin, ray_direction = (
        pixel_to_world_ray(
            u,
            v
        )
    )

    # --------------------------------------------------------
    # Can center height
    # --------------------------------------------------------

    z = can_center_z(
        can["shelf"]
    )

    # --------------------------------------------------------
    # Ray / Z-plane intersection
    # --------------------------------------------------------

    position = ray_intersect_z_plane(
        ray_origin,
        ray_direction,
        z
    )

    can["world_position_m"] = position

    if position is not None:

        can["world_position_mm"] = (
            position * 1000.0
        )

    else:

        can["world_position_mm"] = None


# ============================================================
# PRINT 3D RESULTS
# ============================================================

print()
print("=" * 100)
print("CAN 3D POSITIONS - M53 WORLD FRAME")
print("=" * 100)

print(
    f"{'ID':>3} "
    f"{'Shelf':>5} "
    f"{'Color':>8} "
    f"{'X(mm)':>10} "
    f"{'Y(mm)':>10} "
    f"{'Z(mm)':>10}"
)

print("-" * 100)


for can in cans:

    position = (
        can["world_position_mm"]
    )

    if position is None:

        print(
            f"{can['id']:3d} "
            f"{can['shelf']:5d} "
            f"{can['color']:>8} "
            f"{'ERROR':>10}"
        )

        continue

    x, y, z = position

    print(
        f"{can['id']:3d} "
        f"{can['shelf']:5d} "
        f"{can['color']:>8} "
        f"{x:10.1f} "
        f"{y:10.1f} "
        f"{z:10.1f}"
    )


# ============================================================
# COLOR COUNTS
# ============================================================

counts = Counter(
    c["color"]
    for c in cans
)

print()
print("=" * 70)
print("COLOR COUNTS")
print("=" * 70)

for color in [
    "blue",
    "red",
    "orange",
    "black",
    "green"
]:

    print(
        f"{color:>8}: "
        f"{counts[color]}"
    )


# ============================================================
# DRAW 3D POSITION ON IMAGE
# ============================================================

for can in cans:

    x = can["x"]
    y = can["y"]
    w = can["w"]
    h = can["h"]

    position = (
        can["world_position_mm"]
    )

    # Bounding box
    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    if position is not None:

        X, Y, Z = position

        label = (
            f"{can['id']} "
            f"{can['color']} "
            f"({X:.0f},{Y:.0f},{Z:.0f})"
        )

    else:

        label = (
            f"{can['id']} "
            f"{can['color']} ERROR"
        )

    cv2.putText(
        output,
        label,
        (x - 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    str(OUTPUT_IMAGE),
    output
)

print()
print(
    f"Saved: {OUTPUT_IMAGE}"
)

print()
print(
    f"Total cans: {len(cans)}"
)