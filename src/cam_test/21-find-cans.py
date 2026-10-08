import cv2
import numpy as np
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = Path("/home/mfk01/Pictures/tests/Untitled.png")

OUTPUT_IMAGE = Path("cans_detected.png")

EXPECTED_CANS = 56

# Image dimensions
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 720


# ============================================================
# SHELF REGIONS
# ============================================================
#
# Based on your current camera view.
#
# Each row contains 14 cans.
#
# These values can later be removed when we use the ArUco
# markers to automatically determine shelf positions.
# ============================================================

SHELF_ROWS = [
    (245, 300),   # Shelf 1
    (305, 355),   # Shelf 2
    (360, 405),   # Shelf 3
    (410, 455),   # Shelf 4
]


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise RuntimeError(
        f"Could not load image: {IMAGE_PATH}"
    )

original = image.copy()


# ============================================================
# HSV IMAGE
# ============================================================

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

H, S, V = cv2.split(hsv)


# ============================================================
# COLOR CAN MASK
# ============================================================
#
# Colored cans have:
#
#   S > 20
#   V > 100
#
# This intentionally uses a low saturation threshold because
# the cans in your simulation are pastel colors.
# ============================================================

colored_mask = (
    (S > 20) &
    (V > 100)
).astype(np.uint8) * 255


# Only search the shelf region
colored_mask[:245] = 0
colored_mask[455:] = 0


# ============================================================
# BLACK CAN MASK
# ============================================================
#
# Black cans have low brightness.
# ============================================================

black_mask = (
    (V < 130) &
    (S < 100)
).astype(np.uint8) * 255

black_mask[:245] = 0
black_mask[455:] = 0


# ============================================================
# FIND CAN COMPONENTS
# ============================================================

def find_components(mask, detection_type):

    components = []

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    for i in range(1, num_labels):

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])

        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        area = int(stats[i, cv2.CC_STAT_AREA])

        center_x = float(centroids[i][0])
        center_y = float(centroids[i][1])

        # ----------------------------------------------------
        # Geometry filtering
        # ----------------------------------------------------
        #
        # Your cans appear approximately:
        #
        # width  = 12-18 px
        # height = 23-26 px
        #
        # We allow a larger range for robustness.
        # ----------------------------------------------------

        if area < 80:
            continue

        if area > 600:
            continue

        if w < 8:
            continue

        if w > 25:
            continue

        if h < 18:
            continue

        if h > 35:
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


colored_cans = find_components(
    colored_mask,
    "colored"
)

black_cans = find_components(
    black_mask,
    "black"
)

cans = colored_cans + black_cans


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

    roi_hsv = hsv[
        y:y+h,
        x:x+w
    ]

    roi_H, roi_S, roi_V = cv2.split(
        roi_hsv
    )

    # Use only pixels belonging to the colored object
    valid = (
        (roi_S > 20) &
        (roi_V > 100)
    )

    hue_values = roi_H[valid]

    if len(hue_values) == 0:
        return "unknown"

    # Median is more stable than a single center pixel
    hue = float(
        np.median(hue_values)
    )

    # OpenCV Hue range = 0...179

    if hue < 10 or hue >= 170:
        return "red"

    elif hue < 25:
        return "orange"

    elif hue < 90:
        return "green"

    elif hue < 140:
        return "blue"

    else:
        return "unknown"


# ============================================================
# SHELF LEVEL
# ============================================================

def get_shelf_level(center_y):

    for level, (y_min, y_max) in enumerate(
        SHELF_ROWS,
        start=1
    ):

        if y_min <= center_y <= y_max:
            return level

    return None


# ============================================================
# ASSIGN COLOR + SHELF
# ============================================================

for can in cans:

    can["color"] = classify_color(
        can
    )

    can["shelf"] = get_shelf_level(
        can["center_y"]
    )


# ============================================================
# REMOVE INVALID DETECTIONS
# ============================================================

cans = [
    can
    for can in cans
    if can["shelf"] is not None
    and can["color"] != "unknown"
]


# ============================================================
# SORT
# ============================================================
#
# First by shelf level.
# Then from left -> right.
# ============================================================

cans.sort(
    key=lambda c: (
        c["shelf"],
        c["center_x"]
    )
)


# ============================================================
# ASSIGN CAN IDs
# ============================================================

for index, can in enumerate(
    cans,
    start=1
):

    can["id"] = index


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("CAN DETECTION")
print("=" * 70)

print(
    f"Detected cans: {len(cans)}"
)

print(
    f"Expected cans: {EXPECTED_CANS}"
)

print()

if len(cans) == EXPECTED_CANS:

    print("STATUS: SUCCESS - all 56 cans detected")

else:

    print(
        f"STATUS: WARNING - "
        f"{EXPECTED_CANS - len(cans)} cans missing"
    )


# ============================================================
# SHELF STATISTICS
# ============================================================

print()
print("-" * 70)
print("SHELF COUNTS")
print("-" * 70)

for shelf in range(1, 5):

    shelf_cans = [
        c for c in cans
        if c["shelf"] == shelf
    ]

    print(
        f"Shelf {shelf}: "
        f"{len(shelf_cans)} cans"
    )


# ============================================================
# COLOR STATISTICS
# ============================================================

color_counts = Counter(
    can["color"]
    for can in cans
)

print()
print("-" * 70)
print("COLOR COUNTS")
print("-" * 70)

for color in [
    "blue",
    "red",
    "orange",
    "black",
    "green"
]:

    print(
        f"{color:>8}: "
        f"{color_counts[color]}"
    )


# ============================================================
# DETAILED CAN DATA
# ============================================================

print()
print("-" * 70)
print("CAN DATA")
print("-" * 70)

print(
    f"{'ID':>3} "
    f"{'Shelf':>5} "
    f"{'Color':>8} "
    f"{'X':>8} "
    f"{'Y':>8} "
    f"{'W':>4} "
    f"{'H':>4}"
)

print("-" * 70)

for can in cans:

    print(
        f"{can['id']:3d} "
        f"{can['shelf']:5d} "
        f"{can['color']:>8} "
        f"{can['center_x']:8.1f} "
        f"{can['center_y']:8.1f} "
        f"{can['w']:4d} "
        f"{can['h']:4d}"
    )


# ============================================================
# DRAW DETECTIONS
# ============================================================

output = original.copy()

for can in cans:

    x = can["x"]
    y = can["y"]
    w = can["w"]
    h = can["h"]

    # Bounding box
    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Center
    center = (
        int(can["center_x"]),
        int(can["center_y"])
    )

    cv2.circle(
        output,
        center,
        3,
        (0, 0, 255),
        -1
    )

    # Label
    label = (
        f"{can['id']} "
        f"{can['color']} "
        f"L{can['shelf']}"
    )

    cv2.putText(
        output,
        label,
        (
            x - 5,
            y - 5
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )


# ============================================================
# DRAW TOTAL
# ============================================================

cv2.putText(
    output,
    f"Cans: {len(cans)}/56",
    (20, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2,
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
    f"Saved detection image: {OUTPUT_IMAGE}"
)

print()
print("=" * 70)
