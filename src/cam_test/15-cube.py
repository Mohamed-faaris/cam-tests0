import cv2
import numpy as np
import math


# ============================================================
# SETTINGS
# ============================================================

CAMERA_ID = 1

WARP_SIZE = 360

MIN_FACE_AREA = 7000
MAX_FACE_AREA = 300000

# How strong the grid lines should be
GRID_THRESHOLD = 0.18

# Minimum distance from expected 1/3 and 2/3 grid locations
# before rejecting a candidate.
GRID_TOLERANCE = 0.13

# ============================================================
# COLOR CLASSIFICATION
# ============================================================

def classify_color(hsv_pixel):
    """
    HSV pixel -> Rubik color.
    This is ONLY called after the 3x3 face has been found.
    """

    h, s, v = hsv_pixel

    # White
    if s < 65 and v > 145:
        return "WHITE"

    if v < 45:
        return "UNKNOWN"

    # Red
    if h < 10 or h >= 170:
        return "RED"

    # Orange
    if 10 <= h < 22:
        return "ORANGE"

    # Yellow
    if 22 <= h < 40:
        return "YELLOW"

    # Green
    if 40 <= h < 90:
        return "GREEN"

    # Blue
    if 90 <= h < 135:
        return "BLUE"

    return "UNKNOWN"


# ============================================================
# ORDER FOUR CORNERS
# ============================================================

def order_points(pts):
    """
    Input:
        4 points

    Output:
        top-left
        top-right
        bottom-right
        bottom-left
    """

    pts = np.array(pts, dtype=np.float32)

    result = np.zeros((4, 2), dtype=np.float32)

    # top-left has smallest x+y
    result[0] = pts[np.argmin(pts.sum(axis=1))]

    # bottom-right has largest x+y
    result[2] = pts[np.argmax(pts.sum(axis=1))]

    # top-right has smallest x-y
    result[1] = pts[np.argmin(pts[:, 0] - pts[:, 1])]

    # bottom-left has largest x-y
    result[3] = pts[np.argmax(pts[:, 0] - pts[:, 1])]

    return result


# ============================================================
# DISTANCE
# ============================================================

def point_distance(a, b):
    return np.linalg.norm(a - b)


# ============================================================
# QUADRILATERAL QUALITY
# ============================================================

def is_reasonable_quad(pts):
    """
    Reject obviously bad quadrilaterals.
    """

    pts = order_points(pts)

    tl, tr, br, bl = pts

    width_top = point_distance(tl, tr)
    width_bottom = point_distance(bl, br)

    height_left = point_distance(tl, bl)
    height_right = point_distance(tr, br)

    width = (width_top + width_bottom) / 2
    height = (height_left + height_right) / 2

    if width < 60 or height < 60:
        return False

    ratio = width / height

    # Cube face is approximately square.
    # Perspective can distort it.
    if ratio < 0.45 or ratio > 2.2:
        return False

    return True


# ============================================================
# WARP FACE
# ============================================================

def warp_face(frame, corners):

    destination = np.array(
        [
            [0, 0],
            [WARP_SIZE - 1, 0],
            [WARP_SIZE - 1, WARP_SIZE - 1],
            [0, WARP_SIZE - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    warped = cv2.warpPerspective(
        frame,
        matrix,
        (WARP_SIZE, WARP_SIZE)
    )

    return warped


# ============================================================
# FIND GRID LINES
# ============================================================

def find_grid_score(warped):
    """
    We already assume that 'warped' is a possible cube face.

    Now we look for the internal 3x3 structure.

    We DON'T look at colors.

    We look for strong vertical and horizontal boundaries.
    """

    gray = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )

    # Slight blur reduces noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Edge image
    edges = cv2.Canny(
        gray,
        40,
        120
    )

    h, w = edges.shape

    # Ignore outer ~10%
    margin = int(w * 0.08)

    inner = edges[
        margin:h - margin,
        margin:w - margin
    ]

    # --------------------------------------------------------
    # Vertical projection
    # --------------------------------------------------------

    vertical_strength = np.mean(
        inner,
        axis=0
    ) / 255.0

    # --------------------------------------------------------
    # Horizontal projection
    # --------------------------------------------------------

    horizontal_strength = np.mean(
        inner,
        axis=1
    ) / 255.0

    # --------------------------------------------------------
    # Search around 1/3 and 2/3
    # --------------------------------------------------------

    expected_v1 = w / 3
    expected_v2 = 2 * w / 3

    expected_h1 = h / 3
    expected_h2 = 2 * h / 3

    search_radius = int(
        w * GRID_TOLERANCE
    )

    def strongest_near(signal, expected):

        center = int(expected)

        start = max(
            0,
            center - search_radius
        )

        end = min(
            len(signal),
            center + search_radius
        )

        region = signal[start:end]

        if len(region) == 0:
            return 0, center

        index = np.argmax(region)

        position = start + index

        return signal[position], position

    v1_strength, v1 = strongest_near(
        vertical_strength,
        expected_v1
    )

    v2_strength, v2 = strongest_near(
        vertical_strength,
        expected_v2
    )

    h1_strength, h1 = strongest_near(
        horizontal_strength,
        expected_h1
    )

    h2_strength, h2 = strongest_near(
        horizontal_strength,
        expected_h2
    )

    # --------------------------------------------------------
    # Make sure lines aren't too close together
    # --------------------------------------------------------

    if abs(v2 - v1) < w * 0.15:
        return None

    if abs(h2 - h1) < h * 0.15:
        return None

    # --------------------------------------------------------
    # Calculate grid score
    # --------------------------------------------------------

    vertical_score = (
        v1_strength +
        v2_strength
    ) / 2

    horizontal_score = (
        h1_strength +
        h2_strength
    ) / 2

    grid_score = (
        vertical_score +
        horizontal_score
    ) / 2

    if grid_score < GRID_THRESHOLD:
        return None

    return {
        "score": float(grid_score),
        "v1": int(v1),
        "v2": int(v2),
        "h1": int(h1),
        "h2": int(h2)
    }


# ============================================================
# FIND QUADRILATERALS
# ============================================================

def find_quad_candidates(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Blur
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Canny
    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # Slight dilation joins broken cube edges
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_FACE_AREA:
            continue

        if area > MAX_FACE_AREA:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        # Polygon approximation
        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        points = approx.reshape(
            4,
            2
        ).astype(np.float32)

        if not is_reasonable_quad(points):
            continue

        ordered = order_points(points)

        candidates.append({
            "corners": ordered,
            "area": area
        })

    # Largest first
    candidates.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    return candidates


# ============================================================
# CHECK ONE CANDIDATE
# ============================================================

def evaluate_candidate(frame, candidate):

    corners = candidate["corners"]

    warped = warp_face(
        frame,
        corners
    )

    grid = find_grid_score(
        warped
    )

    if grid is None:
        return None

    return {
        "corners": corners,
        "warped": warped,
        "grid": grid
    }


# ============================================================
# SAMPLE ONE CELL
# ============================================================

def sample_cell(warped, row, col):

    size = WARP_SIZE

    cell_w = size / 3
    cell_h = size / 3

    x1 = int(col * cell_w)
    x2 = int((col + 1) * cell_w)

    y1 = int(row * cell_h)
    y2 = int((row + 1) * cell_h)

    # Don't sample directly at the boundary.
    # Sticker edges can contain shadows/gaps.
    margin_x = int(
        cell_w * 0.25
    )

    margin_y = int(
        cell_h * 0.25
    )

    roi = warped[
        y1 + margin_y:y2 - margin_y,
        x1 + margin_x:x2 - margin_x
    ]

    if roi.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    # Median is more stable than one pixel.
    median = np.median(
        hsv.reshape(-1, 3),
        axis=0
    )

    return classify_color(
        median
    )


# ============================================================
# READ 3x3 COLORS
# ============================================================

def read_colors(warped):

    colors = []

    for row in range(3):

        row_colors = []

        for col in range(3):

            color = sample_cell(
                warped,
                row,
                col
            )

            row_colors.append(
                color
            )

        colors.append(
            row_colors
        )

    return colors


# ============================================================
# DRAW 3x3 GRID
# ============================================================

def draw_grid(frame, corners):

    corners = corners.astype(
        np.int32
    )

    # Outer face
    cv2.polylines(
        frame,
        [corners],
        True,
        (0, 255, 255),
        3
    )

    # We interpolate points along
    # the four edges.

    tl, tr, br, bl = corners

    for i in [1, 2]:

        t = i / 3.0

        # Top -> bottom
        left = (
            tl * (1 - t) +
            bl * t
        )

        right = (
            tr * (1 - t) +
            br * t
        )

        p1 = tuple(
            left.astype(int)
        )

        p2 = tuple(
            right.astype(int)
        )

        cv2.line(
            frame,
            p1,
            p2,
            (0, 255, 255),
            2
        )

        # Left -> right
        top = (
            tl * (1 - t) +
            tr * t
        )

        bottom = (
            bl * (1 - t) +
            br * t
        )

        p1 = tuple(
            top.astype(int)
        )

        p2 = tuple(
            bottom.astype(int)
        )

        cv2.line(
            frame,
            p1,
            p2,
            (0, 255, 255),
            2
        )


# ============================================================
# DRAW COLOR RESULT
# ============================================================

def draw_color_result(frame, colors):

    names = {
        "RED": "R",
        "ORANGE": "O",
        "YELLOW": "Y",
        "GREEN": "G",
        "BLUE": "B",
        "WHITE": "W",
        "UNKNOWN": "?"
    }

    text = []

    for row in colors:

        text.append(
            " ".join(
                names[c]
                for c in row
            )
        )

    y = 95

    for line in text:

        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        y += 30


# ============================================================
# MAIN
# ============================================================

def main():

    cap = cv2.VideoCapture(
        CAMERA_ID
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return

    print(
        "Rubik checkerboard detector started."
    )

    print(
        "Press Q to quit."
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # STEP 1
        # Find geometric quadrilaterals
        # ----------------------------------------------------

        candidates = find_quad_candidates(
            frame
        )

        best = None

        # ----------------------------------------------------
        # STEP 2
        # Test each quad for 3x3 grid
        # ----------------------------------------------------

        for candidate in candidates:

            result = evaluate_candidate(
                frame,
                candidate
            )

            if result is None:
                continue

            if best is None:

                best = result

            else:

                if (
                    result["grid"]["score"]
                    >
                    best["grid"]["score"]
                ):
                    best = result

        # ----------------------------------------------------
        # STEP 3
        # If face found
        # ----------------------------------------------------

        if best is not None:

            corners = best["corners"]

            warped = best["warped"]

            grid = best["grid"]

            # Draw face
            draw_grid(
                frame,
                corners
            )

            # ------------------------------------------------
            # STEP 4
            # NOW read colors
            # ------------------------------------------------

            colors = read_colors(
                warped
            )

            draw_color_result(
                frame,
                colors
            )

            cv2.putText(
                frame,
                f"GRID FOUND  "
                f"{grid['score']:.2f}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "GEOMETRY -> COLOR",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            # Show warped face
            small = cv2.resize(
                warped,
                (180, 180)
            )

            cv2.imshow(
                "Warped Face",
                small
            )

        else:

            cv2.putText(
                frame,
                "SEARCHING FOR 3x3 GRID...",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "COLOR DETECTION DISABLED",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Hide warped window when nothing is found
            blank = np.zeros(
                (180, 180, 3),
                dtype=np.uint8
            )

            cv2.imshow(
                "Warped Face",
                blank
            )

        # ----------------------------------------------------
        # Camera window
        # ----------------------------------------------------

        cv2.imshow(
            "Rubik Cube - Geometry First",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()