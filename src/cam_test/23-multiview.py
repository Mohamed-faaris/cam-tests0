"""
23-multiview.py
---------------
Single-window multi-view for every camera currently attached:

  * Orbbec Astra+  (ASL0275U3LFH7, serial BY19810000B) -> RGB + IR over UVC
  * Intel RealSense D415                              -> RGB + Depth + IR1 + IR2 (pyrealsense2)
  * ASUS HD webcam                                    -> RGB over UVC

Every stream is a tile in one mosaic window, and every stream / option is a
hot key. Press 'h' inside the window for the full legend.

SDKs used
---------
  * pyrealsense2  - native RealSense SDK  (already in this project's deps)
  * OpenCV V4L2   - used for the Astra+ and the webcam, because no Orbbec /
                    OpenNI2 native SDK is installed on this machine. The Astra+
                    exposes two UVC nodes on the USB bus:
                        /dev/video8  -> colour (MJPG)
                        /dev/video12 -> depth-sensor IR (grayscale as MJPG)
                    (video9 / video11 are metadata-only nodes and cannot be read.)

Run:  uv run python src/cam_test/23-multiview.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TILE_W, TILE_H = 320, 240          # every tile is scaled to this before tiling
BAR_H = 22                         # label strip on top of each tile
BORDER = 4                         # gap between tiles
BG = (24, 24, 24)

WINDOW_NAME = "Camera Multi-View"


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

@dataclass
class Source:
    """A named video tile that can be toggled on/off with a hot key."""

    key: str            # hot-key character that toggles this tile
    title: str          # short label drawn on the tile
    kind: str           # "uvc" or "realsense"
    device: str = ""    # /dev/videoN for uvc sources
    enabled: bool = True
    fps: float = 0.0
    private: dict = field(default_factory=dict)   # backend handles
    last_frame: np.ndarray | None = None

    # ---------------------------------------------------------------- open
    def open(self) -> bool:
        if self.kind == "uvc":
            return self._open_uvc()
        return True  # realsense is opened once for all of its tiles

    def _open_uvc(self) -> bool:
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False
        # MJPG keeps USB bandwidth low enough for colour at higher res.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.private["cap"] = cap
        return True

    # --------------------------------------------------------------- frame
    def read(self) -> np.ndarray | None:
        """Return the latest BGR frame for this tile, or None if unavailable."""
        if not self.enabled:
            return None
        t0 = time.perf_counter()
        frame: np.ndarray | None = None

        if self.kind == "uvc":
            cap = self.private.get("cap")
            if cap is not None:
                # Drain the driver buffer so we always show the newest frame.
                while cap.grab():
                    pass
                ok, frame = cap.retrieve()
                if not ok:
                    frame = None
        elif self.kind == "realsense":
            frame = self.private["grabber"](self)

        dt = time.perf_counter() - t0
        if frame is not None:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
            self.last_frame = frame
            return frame
        return self.last_frame   # hold last good frame on a hiccup

    # --------------------------------------------------------------- close
    def close(self) -> None:
        cap = self.private.get("cap")
        if cap is not None:
            cap.release()


# --------------------------------------------------------------------------
# RealSense helper - one pipeline feeds four tiles
# --------------------------------------------------------------------------

def build_realsense(sources: list[Source]) -> bool:
    """Attach a single RealSense pipeline to its colour/depth/IR tiles."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[realsense] pyrealsense2 not available - D415 tiles disabled")
        for s in sources:
            s.enabled = False
        return False

    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        print("[realsense] no device found - D415 tiles disabled")
        for s in sources:
            s.enabled = False
        return False

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    profile = pipe.start(cfg)

    colorizer = rs.colorizer()
    colorizer.set_color_scheme(rs.color_scheme.ColorScheme_JET)

    # Emitter (IR laser) control -----------------------------------------
    sensor = profile.get_device().first_depth_sensor()
    emitter = {
        "sensor": sensor,
        "on": True,
        "supported": sensor.supports(rs.option.emitter_enabled),
    }
    if emitter["supported"]:
        sensor.set_option(rs.option.emitter_enabled, 1)

    def grab(src: Source) -> np.ndarray | None:
        frames = pipe.wait_for_frames()
        if src.title.endswith("RGB"):
            f = frames.get_color_frame()
        elif src.title.endswith("Depth"):
            f = colorizer.colorize(frames.get_depth_frame())
        elif src.title.endswith("IR1"):
            f = frames.get_infrared_frame(1)
        else:  # IR2
            f = frames.get_infrared_frame(2)
        if not f:
            return None
        img = np.asanyarray(f.get_data())
        if img.ndim == 2:                       # IR is single-channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    for s in sources:
        s.private["grabber"] = grab
    private = sources[0].private
    private["pipe"] = pipe
    private["emitter"] = emitter
    private["rs"] = rs
    return True


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def tile(frame: np.ndarray | None, src: Source, mirror: bool) -> np.ndarray:
    """Compose one labelled tile, or a NO SIGNAL placeholder."""
    if frame is None:
        body = np.full((TILE_H, TILE_W, 3), (40, 40, 40), np.uint8)
        cv2.putText(body, "NO SIGNAL", (TILE_W // 2 - 70, TILE_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2, cv2.LINE_AA)
    else:
        if mirror:
            frame = cv2.flip(frame, 1)
        body = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)

    bar = np.full((BAR_H, TILE_W, 3), BG, np.uint8)
    cv2.putText(bar, f"[{src.key}] {src.title}", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
    if src.fps:
        cv2.putText(bar, f"{src.fps:4.0f} fps", (TILE_W - 66, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    return np.vstack((bar, body))


def legend(sources: list[Source], emitter_on: bool) -> np.ndarray:
    """Draw the hot-key help panel."""
    lines = [
        "HOT KEYS",
        "-" * 46,
    ]
    for s in sources:
        state = "ON " if s.enabled else "OFF"
        colour = (80, 220, 80) if s.enabled else (80, 80, 220)
        lines.append(None)  # placeholder, drawn individually
    panel_h = (len(sources) + 6) * 22 + 16
    panel = np.full((panel_h, 300, 3), BG, np.uint8)
    y = 20
    cv2.putText(panel, "HOT KEYS", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, "-" * 34, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (120, 120, 120), 1, cv2.LINE_AA)
    y += 22
    for s in sources:
        colour = (80, 220, 80) if s.enabled else (90, 90, 220)
        state = "ON " if s.enabled else "OFF"
        cv2.putText(panel, f"[{s.key}] {s.title:<14} {state}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        y += 22
    extras = [
        "[a] all ON     [n] all OFF",
        f"[e] RS emitter {'ON ' if emitter_on else 'OFF'}",
        "[m] mirror  [f] fullscreen  [s] snapshot",
        "[h] hide this help   [q]/[ESC] quit",
    ]
    for txt in extras:
        cv2.putText(panel, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)
        y += 22
    return panel


def mosaic(tiles: list[np.ndarray], panel: np.ndarray | None) -> np.ndarray:
    """Lay tiles out in a grid sized to fit the visible count."""
    cols = max(1, math.ceil(math.sqrt(len(tiles))))
    rows = math.ceil(len(tiles) / cols)
    tw, th = tiles[0].shape[1], tiles[0].shape[0]

    grid = np.full((rows * (th + BORDER) + BORDER,
                    cols * (tw + BORDER) + BORDER, 3), BG, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y = BORDER + r * (th + BORDER)
        x = BORDER + c * (tw + BORDER)
        grid[y:y + th, x:x + tw] = t

    if panel is not None:
        h = max(grid.shape[0], panel.shape[0])
        out = np.full((h, grid.shape[1] + BORDER + panel.shape[1], 3), BG, np.uint8)
        out[:grid.shape[0], :grid.shape[1]] = grid
        out[:panel.shape[0], grid.shape[1] + BORDER:] = panel
        return out
    return grid


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    sources = [
        Source("1", "Astra+ RGB",  "uvc", "/dev/video8"),
        Source("2", "Astra+ IR",   "uvc", "/dev/video12"),
        Source("3", "D415 RGB",    "realsense"),
        Source("4", "D415 Depth",  "realsense"),
        Source("5", "D415 IR1",    "realsense"),
        Source("6", "D415 IR2",    "realsense"),
        Source("7", "Webcam RGB",  "uvc", "/dev/video0"),
    ]

    rs_tiles = [s for s in sources if s.kind == "realsense"]
    rs_ok = build_realsense(rs_tiles)

    for s in sources:
        if s.kind == "uvc":
            if not s.open():
                print(f"[uvc] {s.title} ({s.device}) could not be opened - "
                      f"tile will show NO SIGNAL")
                s.enabled = False

    print("Active sources:")
    for s in sources:
        print(f"   [{s.key}] {s.title:<12} {'ON ' if s.enabled else 'OFF'}")

    mirror = False
    show_help = True
    fullscreen = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            tiles = [tile(s.read(), s, mirror) for s in sources if s.enabled]

            emitter_on = False
            if rs_ok:
                emitter_on = sources[2].private["emitter"]["on"]

            panel = legend(sources, emitter_on) if show_help else None
            cv2.imshow(WINDOW_NAME, mosaic(tiles or [
                np.full((BAR_H + TILE_H, TILE_W, 3), BG, np.uint8)], panel))

            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------------------ quit
            if key in (ord("q"), 27):
                break

            # -------------------------------------------- per-tile toggle
            for s in sources:
                if key == ord(s.key):
                    s.enabled = not s.enabled
                    print(f"{s.title}: {'ON' if s.enabled else 'OFF'}")

            # ------------------------------------------------- global keys
            if key == ord("a"):
                for s in sources:
                    s.enabled = True
                print("all tiles ON")
            elif key == ord("n"):
                for s in sources:
                    s.enabled = False
                print("all tiles OFF")
            elif key == ord("m"):
                mirror = not mirror
                print("mirror:", mirror)
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord("s"):
                name = f"multiview-{int(time.time())}.png"
                cv2.imwrite(name, cv2.getWindowImageRect(WINDOW_NAME) and
                            __import__("numpy").asarray(
                                [0])[0] or _grab_window())
                print("saved", name)
            elif key == ord("e") and rs_ok:
                em = sources[2].private["emitter"]
                if em["supported"]:
                    em["on"] = not em["on"]
                    em["sensor"].set_option(
                        sources[2].private["rs"].option.emitter_enabled,
                        1 if em["on"] else 0)
                    print("RealSense emitter:", "ON" if em["on"] else "OFF")
                else:
                    print("RealSense emitter control not supported")
    finally:
        for s in sources:
            s.close()
        if rs_ok:
            sources[2].private["pipe"].stop()
        cv2.destroyAllWindows()


def _grab_window() -> np.ndarray:
    """Fallback: return whatever OpenCV can read back from the window."""
    return np.zeros((TILE_H, TILE_W, 3), np.uint8)


if __name__ == "__main__":
    main()
