"""
27-april-tag-identifier.py
--------------------------
Live AprilTag (tag36h11) detector: draws a box around each tag and prints its ID.

Uses OpenCV's ArUco module with the AprilTag dictionary.
No extra detector library needed.

Run:  uv run python src/cam_test/27-april-tag-identifier.py
      uv run python src/cam_test/27-april-tag-identifier.py --image apriltags_all_layouts.png
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 36h11 = 36 data bits, 11-bit Hamming correction
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11

BOX_COLOR = (0, 255, 0)      # green box
TEXT_COLOR = (0, 255, 255)   # yellow ID text


# ============================================================
# DETECTOR
# ============================================================

detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(TAG_FAMILY),
    detector_params,
)


def detect(gray: np.ndarray):
    """Return (corners, ids) for every tag36h11 tag in the frame."""
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def tag_ids_to_list(ids) -> list[int]:
    """Flatten the ids array (shape (N,) or (N, 1)) to a plain list."""
    if ids is None:
        return []
    return [int(i) for i in np.asarray(ids).ravel()]


def draw(display: np.ndarray, corners, ids) -> None:
    """Box each tag and label it with its ID."""
    if ids is None:
        return

    cv2.aruco.drawDetectedMarkers(display, corners, ids, BOX_COLOR)

    for corner, tag_id in zip(corners, tag_ids_to_list(ids)):
        # Top-left corner of the tag, offset up so the text clears the box
        x = int(corner[0][0][0])
        y = int(corner[0][0][1])

        cv2.putText(
            display,
            f"ID {tag_id}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2,
        )


def overlay_status(display: np.ndarray, count: int) -> None:
    color = BOX_COLOR if count else (0, 0, 255)
    label = f"TAGS: {count}" if count else "NO TAGS"
    cv2.putText(
        display,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )
    cv2.putText(
        display,
        "ESC = exit",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


# ============================================================
# IMAGE MODE
# ============================================================

def run_image(path: str) -> None:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image:\n{path}")

    corners, ids = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    draw(frame, corners, ids)

    count = len(tag_ids_to_list(ids))
    overlay_status(frame, count)

    print(f"{count} tag(s) found")
    print("IDs:", ", ".join(str(i) for i in tag_ids_to_list(ids)) or "-")

    cv2.imshow("AprilTag 36h11", frame)
    while True:
        if cv2.waitKey(33) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


# ============================================================
# LIVE MODE
# ============================================================

def run_live() -> None:
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nCamera read failed.")
                break

            corners, ids = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            draw(frame, corners, ids)

            found_ids = tag_ids_to_list(ids)
            overlay_status(frame, len(found_ids))

            found = ", ".join(str(i) for i in found_ids)
            print(f"\rTAGS: {len(found_ids)}  IDs: {found or '-':<20}", end="", flush=True)

            cv2.imshow("AprilTag 36h11", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        if idx + 1 >= len(sys.argv):
            raise SystemExit("Usage: --image <path>")
        run_image(sys.argv[idx + 1])
        return

    run_live()


if __name__ == "__main__":
    main()
