"""
25-astra-test.py
----------------
Simple Astra+ camera + sensor inventory using only OpenCV (no SDK needed).

Modes:
  (default)  enumerate + capture one frame per node -> astra_captures/
  --view     load saved captures and display them side-by-side in OpenCV
  --live     open every readable Astra node and stream live in OpenCV

Run:  uv run python src/cam_test/25-astra-test.py
      uv run python src/cam_test/25-astra-test.py --view
      uv run python src/cam_test/25-astra-test.py --live
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_DIR = "/sys/class/video4linux"
OUT_DIR = Path("astra_captures")
OUT_DIR.mkdir(exist_ok=True)


def read_sysfs(node: str, field: str) -> str:
    """Read a sysfs field for a video4linux node, or '' if missing."""
    path = os.path.join(VIDEO_DIR, node, field)
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def v4l2_ctl(node: str, *args: str) -> str:
    """Run v4l2-ctl if available, else return ''."""
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/{node}", *args],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def list_cameras() -> list[str]:
    """Return sorted video4linux node names (video0, video1, ...)."""
    if not os.path.isdir(VIDEO_DIR):
        return []
    nodes = [n for n in os.listdir(VIDEO_DIR) if n.startswith("video")]
    return sorted(nodes, key=lambda n: int(n.replace("video", "")))


def is_astra(node: str) -> bool:
    """True if the node belongs to the Astra+ (Astrap / Orbbec label)."""
    label = read_sysfs(node, "name").lower()
    return "astrap" in label or "orbbec" in label


def open_astra_caps(nodes: list[str]) -> list[tuple[str, cv2.VideoCapture]]:
    """Open every readable Astra node, return (label, cap) pairs."""
    caps: list[tuple[str, cv2.VideoCapture]] = []
    for node in nodes:
        dev = f"/dev/{node}"
        label = read_sysfs(node, "name") or node
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            print(f"[{node}] {label}: could not open")
            continue
        # MJPG keeps USB bandwidth low enough for colour at higher res.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        caps.append((f"{node} {label}", cap))
    return caps


def live_stream() -> None:
    """Open every readable Astra node and stream live in OpenCV."""
    nodes = [n for n in list_cameras() if is_astra(n)]
    if not nodes:
        print("No Astra+ / Orbbec video devices found.")
        return

    caps = open_astra_caps(nodes)
    if not caps:
        print("No Astra nodes could be opened.")
        return

    print(f"Streaming {len(caps)} Astra source(s). Press Q or ESC to quit.")
    try:
        while True:
            tiles: list[tuple[str, np.ndarray]] = []
            for label, cap in caps:
                ok, frame = cap.read()
                if ok and frame is not None:
                    tiles.append((label, frame))
                else:
                    tiles.append((label, np.zeros((480, 640, 3), np.uint8)))

            canvas = stack_tiles(tiles)
            cv2.imshow("Astra+ live stream", canvas)
            key = cv2.waitKey(33) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for _, cap in caps:
            cap.release()
        cv2.destroyAllWindows()


def stack_tiles(tiles: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Compose labelled tiles into a single canvas."""
    BAR_H = 26
    BG = (24, 24, 24)
    target_h = 480
    parts: list[np.ndarray] = []
    for label, img in tiles:
        h, w = img.shape[:2]
        new_w = int(w * target_h / h)
        img = cv2.resize(img, (new_w, target_h))
        bar = np.full((BAR_H, img.shape[1], 3), BG, np.uint8)
        cv2.putText(bar, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 255, 200), 1, cv2.LINE_AA)
        parts.append(np.vstack((bar, img)))
    return np.hstack(parts)


def view_captures() -> None:
    """Load saved PNGs from OUT_DIR and show them side-by-side in OpenCV."""
    files = sorted(OUT_DIR.glob("video*.png"))
    if not files:
        print(f"No captures found in {OUT_DIR}/. Run without --view first.")
        return

    images: list[tuple[str, np.ndarray]] = []
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            images.append((f.stem, img))

    if not images:
        print("No readable images.")
        return

    # Scale every image to a common height for side-by-side display.
    target_h = 480
    tiles: list[tuple[str, np.ndarray]] = []
    for name, img in images:
        h, w = img.shape[:2]
        new_w = int(w * target_h / h)
        tiles.append((name, cv2.resize(img, (new_w, target_h))))

    # Stack horizontally with a label bar on top of each.
    BAR_H = 26
    BG = (24, 24, 24)
    bars: list[np.ndarray] = []
    for (name, img) in tiles:
        bar = np.full((BAR_H, img.shape[1], 3), BG, np.uint8)
        cv2.putText(bar, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 255, 200), 1, cv2.LINE_AA)
        bars.append(bar)
        tiles[tiles.index((name, img))] = (name, np.vstack((bar, img)))

    canvas = np.hstack([t for _, t in tiles])
    cv2.imshow("Astra+ Captures (press Q or ESC to quit)", canvas)
    print("Showing captures. Press Q or ESC in the window to close.")

    while True:
        key = cv2.waitKey(33) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main() -> None:
    if "--live" in sys.argv:
        live_stream()
        return
    if "--view" in sys.argv:
        view_captures()
        return

    nodes = [n for n in list_cameras() if is_astra(n)]
    if not nodes:
        print("No Astra+ / Orbbec video devices found.")
        return

    print(f"Found {len(nodes)} video device(s):\n")

    for node in nodes:
        dev = f"/dev/{node}"
        label = read_sysfs(node, "name") or "(unknown)"
        idx = read_sysfs(node, "index")

        print(f"[{node}] {label}  (index={idx})")

        # Sensor info from sysfs
        for field in ("driver", "card", "bus_info"):
            val = read_sysfs(node, field)
            if val:
                print(f"    {field:<10}: {val}")

        # Capabilities via v4l2-ctl (best-effort)
        caps = v4l2_ctl(node, "--list-formats")
        if caps:
            print("    formats:")
            for line in caps.splitlines()[1:]:
                print(f"        {line.strip()}")

        # Open with OpenCV and grab one frame
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            # Try without the backend hint
            cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            print("    capture: FAILED to open\n")
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))

        print(f"    default: {w}x{h}  {fourcc_str}  {fps:.1f} fps")

        # Grab a single frame
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = cap.read()
        cap.release()

        if ok and frame is not None:
            out = OUT_DIR / f"{node}.png"
            cv2.imwrite(str(out), frame)
            print(f"    saved:   {out}  ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print("    capture: no frame returned")
        print()


if __name__ == "__main__":
    main()