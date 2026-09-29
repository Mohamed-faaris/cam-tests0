import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BOARD_SIZE = (8, 5)
SQUARE_SIZE_MM = 4.0

# Minimum contour area in pixels.
# Increase this if small noise is detected.
MIN_OBJECT_AREA = 500

# Number of objects we want
MAX_OBJECTS = 2


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

data = np.load("camera_calibration.npz")

camera_matrix = data["camera_matrix"]
distortion = data["distortion"]


# ============================================================
# CHECKERBOARD 3D POINTS
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
# PIXEL -> WORLD
# ============================================================

def pixel_to_world(pixel, rvec, tvec):

    R, _ = cv2.Rodrigues(rvec)

    # Pixel -> normalized camera coordinates
    p = np.array(
        [[[float(pixel[0]), float(pixel[1])]]],
        dtype=np.float64
    )

    undistorted = cv2.undistortPoints(
        p,
        camera_matrix,
        distortion
    )

    ray_camera = np.array(
        [
            undistorted[0, 0, 0],
            undistorted[0, 0, 1],
            1.0
        ],
        dtype=np.float64
    )

    # Camera -> world
    R_inv = R.T

    camera_position_world = (
        -R_inv @ tvec.reshape(3)
    )

    ray_world = R_inv @ ray_camera

    # Intersection with Z = 0
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

    return world_point[:2]


# ============================================================
# CONTOUR -> WORLD POLYGON
# ============================================================

def contour_to_world(contour, rvec, tvec):

    world_points = []

    for point in contour:

        x, y = point[0]

        world = pixel_to_world(
            (x, y),
            rvec,
            tvec
        )

        if world is not None:
            world_points.append(world)

    if len(world_points) == 0:
        return None

    return np.array(world_points)


# ============================================================
# DISTANCE BETWEEN TWO POLYGONS
# ============================================================

def polygon_distance(poly_a, poly_b):

    min_distance = float("inf")

    closest_a = None
    closest_b = None

    # Compare every edge of A with every edge of B.
    #
    # For the first prototype, we use point-to-segment
    # distance for every polygon vertex.

    def point_to_segment_distance(p, a, b):

        ab = b - a

        length_squared = np.dot(ab, ab)

        if length_squared == 0:
            return np.linalg.norm(p - a), a

        t = np.dot(p - a, ab) / length_squared

        t = np.clip(t, 0.0, 1.0)

        projection = a + t * ab

        distance = np.linalg.norm(p - projection)

        return distance, projection

    # Vertices A -> edges B
    for p in poly_a:

        for i in range(len(poly_b)):

            a = poly_b[i]
            b = poly_b[(i + 1) % len(poly_b)]

            distance, closest = point_to_segment_distance(
                p,
                a,
                b
            )

            if distance < min_distance:

                min_distance = distance
                closest_a = p
                closest_b = closest

    # Vertices B -> edges A
    for p in poly_b:

        for i in range(len(poly_a)):

            a = poly_a[i]
            b = poly_a[(i + 1) % len(poly_a)]

            distance, closest = point_to_segment_distance(
                p,
                a,
                b
            )

            if distance < min_distance:

                min_distance = distance
                closest_a = closest
                closest_b = p

    return (
        min_distance,
        closest_a,
        closest_b
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera read failed")
        break

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

    if not found:

        cv2.putText(
            frame,
            "CHECKERBOARD NOT FOUND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "Object Distance",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue


    # ========================================================
    # ESTIMATE CHECKERBOARD POSE
    # ========================================================

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        continue


    # ========================================================
    # DRAW CHECKERBOARD
    # ========================================================

    cv2.drawChessboardCorners(
        frame,
        BOARD_SIZE,
        corners,
        found
    )


    # ========================================================
    # CREATE OBJECT MASK
    # ========================================================

    # Threshold dark objects.
    #
    # Objects should be significantly darker than the
    # background/checkerboard.

    _, mask = cv2.threshold(
        gray,
        80,
        255,
        cv2.THRESH_BINARY_INV
    )


    # ========================================================
    # REMOVE CHECKERBOARD AREA
    # ========================================================

    checkerboard_image_points = corners.reshape(-1, 2)

    hull = cv2.convexHull(
        checkerboard_image_points.astype(np.float32)
    )

    checkerboard_mask = np.zeros_like(gray)

    cv2.fillConvexPoly(
        checkerboard_mask,
        hull.astype(np.int32),
        255
    )

    # Objects only inside checkerboard region
    mask = cv2.bitwise_and(
        mask,
        checkerboard_mask
    )


    # ========================================================
    # CLEAN MASK
    # ========================================================

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # ========================================================
    # FILTER CONTOURS
    # ========================================================

    object_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_OBJECT_AREA:
            continue

        object_contours.append(
            contour
        )


    # Largest first
    object_contours.sort(
        key=cv2.contourArea,
        reverse=True
    )

    object_contours = object_contours[
        :MAX_OBJECTS
    ]


    # ========================================================
    # DRAW OBJECTS
    # ========================================================

    world_polygons = []

    centers = []


    for index, contour in enumerate(object_contours):

        # ----------------------------------------------------
        # Draw contour
        # ----------------------------------------------------

        cv2.drawContours(
            frame,
            [contour],
            -1,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Center using image moments
        # ----------------------------------------------------

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = (
            moments["m10"]
            / moments["m00"]
        )

        cy = (
            moments["m01"]
            / moments["m00"]
        )

        center_pixel = (
            int(cx),
            int(cy)
        )

        # ----------------------------------------------------
        # Convert center to world
        # ----------------------------------------------------

        center_world = pixel_to_world(
            center_pixel,
            rvec,
            tvec
        )

        if center_world is None:
            continue

        X = center_world[0]
        Y = center_world[1]

        centers.append(
            center_world
        )

        # ----------------------------------------------------
        # Convert entire contour
        # ----------------------------------------------------

        world_polygon = contour_to_world(
            contour,
            rvec,
            tvec
        )

        if world_polygon is None:
            continue

        world_polygons.append(
            world_polygon
        )

        # ----------------------------------------------------
        # Draw center
        # ----------------------------------------------------

        cv2.circle(
            frame,
            center_pixel,
            7,
            (255, 0, 255),
            -1
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"OBJ {index + 1}: "
            f"({X:.1f}, {Y:.1f}) mm"
        )

        cv2.putText(
            frame,
            label,
            (
                center_pixel[0] - 80,
                center_pixel[1] - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # ========================================================
    # TWO OBJECTS
    # ========================================================

    if len(world_polygons) == 2:

        poly_a = world_polygons[0]
        poly_b = world_polygons[1]

        # ----------------------------------------------------
        # Center distance
        # ----------------------------------------------------

        center_distance = np.linalg.norm(
            centers[0] - centers[1]
        )

        # ----------------------------------------------------
        # Edge distance
        # ----------------------------------------------------

        edge_distance, closest_a, closest_b = (
            polygon_distance(
                poly_a,
                poly_b
            )
        )

        # ----------------------------------------------------
        # Collision
        # ----------------------------------------------------

        collision = edge_distance < 1.0

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Center distance: "
            f"{center_distance:.2f} mm",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Edge distance: "
            f"{edge_distance:.2f} mm",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Collision: "
            f"{'YES' if collision else 'NO'}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if collision else (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Draw closest points
        # ----------------------------------------------------

        pa = tuple(
            np.round(closest_a).astype(int)
        )

        pb = tuple(
            np.round(closest_b).astype(int)
        )

        # We need to project world points back to image
        # for visualization.

        points_world = np.array(
            [
                [closest_a[0], closest_a[1], 0],
                [closest_b[0], closest_b[1], 0]
            ],
            dtype=np.float32
        )

        projected, _ = cv2.projectPoints(
            points_world,
            rvec,
            tvec,
            camera_matrix,
            distortion
        )

        projected = projected.reshape(-1, 2)

        p1 = tuple(
            projected[0].astype(int)
        )

        p2 = tuple(
            projected[1].astype(int)
        )

        cv2.line(
            frame,
            p1,
            p2,
            (255, 0, 255),
            3
        )

        cv2.circle(
            frame,
            p1,
            6,
            (255, 0, 255),
            -1
        )

        cv2.circle(
            frame,
            p2,
            6,
            (255, 0, 255),
            -1
        )


    # ========================================================
    # STATUS
    # ========================================================

    cv2.putText(
        frame,
        f"Objects detected: "
        f"{len(object_contours)}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "ESC = exit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Object Distance",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()