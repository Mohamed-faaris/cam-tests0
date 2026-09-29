import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

CAMERA_ID = 0

# Objects we are looking for
MIN_AREA = 20
MAX_AREA = 3000

# Reject very large filled regions
MAX_WIDTH = 100
MAX_HEIGHT = 100

# How thin a region can be
MIN_ASPECT_RATIO = 1.05

# Morphology
KERNEL_SIZE = 3


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Slight blur
    # --------------------------------------------------------

    gray_blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # Threshold dark objects
    # --------------------------------------------------------

    _, mask = cv2.threshold(
        gray_blur,
        100,
        255,
        cv2.THRESH_BINARY_INV
    )

    # --------------------------------------------------------
    # Remove tiny noise
    # --------------------------------------------------------

    kernel = np.ones(
        (KERNEL_SIZE, KERNEL_SIZE),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    # ========================================================
    # ANALYZE EACH CONTOUR
    # ========================================================

    for contour in contours:

        area = cv2.contourArea(contour)

        # ----------------------------------------------------
        # Area filter
        # ----------------------------------------------------

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x, y, w, h = cv2.boundingRect(contour)

        if w > MAX_WIDTH:
            continue

        if h > MAX_HEIGHT:
            continue

        # ----------------------------------------------------
        # Aspect ratio
        # ----------------------------------------------------

        aspect = max(w, h) / max(
            min(w, h),
            1
        )

        # ----------------------------------------------------
        # Solidity
        #
        # Filled checkerboard squares have high solidity.
        # Thin irregular objects tend to have lower solidity.
        # ----------------------------------------------------

        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            continue

        solidity = area / hull_area

        # ----------------------------------------------------
        # Extent
        #
        # How much of the bounding box is filled.
        # ----------------------------------------------------

        bounding_area = w * h

        if bounding_area == 0:
            continue

        extent = area / bounding_area

        # ----------------------------------------------------
        # Perimeter / area
        #
        # Thin irregular shapes have relatively large
        # perimeter compared with their area.
        # ----------------------------------------------------

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        compactness = (
            perimeter * perimeter
            / (4 * np.pi * area)
        )

        # ----------------------------------------------------
        # Candidate
        # ----------------------------------------------------

        # We want irregular objects rather than solid squares.
        #
        # Don't make these filters too strict yet.

        if extent > 0.75:
            continue

        candidates.append({
            "contour": contour,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "solidity": solidity,
            "extent": extent,
            "compactness": compactness
        })


    # ========================================================
    # SCORE CANDIDATES
    # ========================================================

    # The target in your image is relatively small and
    # irregular. Rank candidates by compactness.

    candidates.sort(
        key=lambda c: c["compactness"],
        reverse=True
    )


    # ========================================================
    # DRAW RESULTS
    # ========================================================

    for i, candidate in enumerate(candidates):

        contour = candidate["contour"]

        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

        # ----------------------------------------------------
        # Center
        # ----------------------------------------------------

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = int(
            moments["m10"]
            / moments["m00"]
        )

        cy = int(
            moments["m01"]
            / moments["m00"]
        )

        # ----------------------------------------------------
        # Draw contour
        # ----------------------------------------------------

        cv2.drawContours(
            frame,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Center point
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        text = (
            f"OBJ {i + 1} "
            f"({cx},{cy})"
        )

        cv2.putText(
            frame,
            text,
            (x, max(y - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"Candidates: {len(candidates)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW MASK
    # ========================================================

    cv2.imshow(
        "Object Detection",
        frame
    )

    cv2.imshow(
        "Threshold",
        mask
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()