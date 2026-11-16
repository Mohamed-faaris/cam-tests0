#!/usr/bin/env python3
"""Capture registered RGB-D data from an Intel RealSense camera.

Uses librealsense/pyrealsense2 alignment (depth -> color). Saves:
  rgb/                    BGR8 PNG frames
  depth/                  aligned float32 NumPy arrays (metres; WAPR-compatible)
  depth_aligned_mm/       aligned uint16 PNGs (millimetres; color pixel grid)
  depth_raw_z16/          original sensor Z16 PNGs (native depth pixel grid)
  depth_preview/          8-bit grayscale and colorized aligned-depth previews
  cam_K.txt               RGB camera intrinsics, for WAPR
  camera_metadata.json    stream profiles, intrinsics, extrinsics, timestamps, etc.
  recording.bag           optional original sensor recording (--record-bag)

Run `python3 realsense_rgbd_capture.py --help` for options.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_realsense():
    try:
        import pyrealsense2 as rs  # type: ignore
        return rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not available in this Python interpreter.\n"
            "Install/use the librealsense Python bindings that match this Python, then retry.\n"
            "Check with: python3 -c 'import pyrealsense2 as rs; print(rs.__version__)'"
        ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture RGB-D with RealSense SDK depth-to-color alignment.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory (default: ./realsense_capture_YYYYmmdd_HHMMSS)")
    p.add_argument("--serial", default=None, help="RealSense device serial number")
    p.add_argument("--profile", choices=("auto", "1280x720@30", "848x480@30", "640x480@30", "640x360@30"),
                   default="auto", help="Preferred shared stream profile; auto tries the highest preferred profile first")
    p.add_argument("--frames", type=int, default=0, help="Stop after N frames (0 = until q/Esc/Ctrl-C)")
    p.add_argument("--seconds", type=float, default=0,
                   help="Stop after this many seconds (0 = until q/Esc/Ctrl-C)")
    p.add_argument("--record-bag", action="store_true",
                   help="Also save the original camera streams to recording.bag (raw streams; align again on playback)")
    p.add_argument("--no-preview", action="store_true", help="Do not open the live OpenCV preview window")
    p.add_argument("--preview-near", type=float, default=0.20, help="Near limit for depth preview, metres")
    p.add_argument("--preview-far", type=float, default=2.00, help="Far limit for depth preview, metres")
    p.add_argument("--max-sync-ms", type=float, default=30.0,
                   help="Flag frame pairs whose SDK depth/color timestamp difference exceeds this value")
    return p.parse_args()


def stream_profiles():
    # Try the preferred D415 common profiles first. Unsupported profiles fall back cleanly.
    return {
        "auto": [(1280, 720, 30), (848, 480, 30), (640, 480, 30), (640, 360, 30)],
        "1280x720@30": [(1280, 720, 30)],
        "848x480@30": [(848, 480, 30)],
        "640x480@30": [(640, 480, 30)],
        "640x360@30": [(640, 360, 30)],
    }


def intrinsics_dict(intr: Any) -> dict[str, Any]:
    return {
        "width": int(intr.width), "height": int(intr.height),
        "fx": float(intr.fx), "fy": float(intr.fy),
        "ppx": float(intr.ppx), "ppy": float(intr.ppy),
        "model": str(intr.model), "coeffs": [float(x) for x in intr.coeffs],
        "K": [[float(intr.fx), 0.0, float(intr.ppx)],
              [0.0, float(intr.fy), float(intr.ppy)],
              [0.0, 0.0, 1.0]],
    }


def profile_dict(profile: Any) -> dict[str, Any]:
    v = profile.as_video_stream_profile()
    return {
        "stream": str(profile.stream_type()), "format": str(profile.format()),
        "width": int(v.width()), "height": int(v.height()), "fps": int(profile.fps()),
        "intrinsics": intrinsics_dict(v.get_intrinsics()),
    }


def depth_preview(depth_m: np.ndarray, near_m: float, far_m: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(depth_m) & (depth_m >= near_m) & (depth_m <= far_m)
    gray = np.zeros(depth_m.shape, dtype=np.uint8)
    gray[valid] = np.clip((depth_m[valid] - near_m) / (far_m - near_m) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return gray, color


def mkdirs(root: Path) -> dict[str, Path]:
    names = ("rgb", "depth", "depth_aligned_mm", "depth_raw_z16", "depth_preview")
    folders = {name: root / name for name in names}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=False)
    return folders


def write_conversion_json(out: Path, metadata: dict[str, Any], records: list[dict[str, Any]],
                          max_sync_ms: float, bag_path: Path | None) -> None:
    """Write the compact conversion.json schema used by the WAPR dataset tools."""
    frames = [{
        "frame": row["frame"],
        "rgb_timestamp_ns": int(round(row["rgb_timestamp_ms"] * 1_000_000)),
        "depth_timestamp_ns": int(round(row["depth_timestamp_ms"] * 1_000_000)),
        "sync_delta_ms": float(row["sync_delta_ms"]),
    } for row in records]
    conversion = {
        "source": "RealSense SDK live capture",
        "bag": str(bag_path) if bag_path else None,
        "device_serial": metadata.get("device", {}).get("serial"),
        "rgb_topic": "RealSense color stream",
        "depth_topic": "RealSense depth stream aligned to color by librealsense",
        "camera_info_topic": "RealSense color stream intrinsics",
        "depth_scale_16uc1_m": metadata.get("depth_scale_m_per_unit"),
        "depth_units_saved": "float32 metres in depth/<frame>.npy",
        "depth_registered_to": "color camera via pyrealsense2.align(rs.stream.color)",
        "max_sync_ms": float(max_sync_ms),
        "paired_frames": len(frames),
        "frames": frames,
    }
    (out / "conversion.json").write_text(json.dumps(conversion, indent=2) + "\n")


def start_pipeline(rs: Any, serial: str | None, profile_choice: str, bag_path: Path | None):
    errors: list[str] = []
    for width, height, fps in stream_profiles()[profile_choice]:
        pipeline = rs.pipeline()
        config = rs.config()
        if bag_path is not None:
            # A failed profile attempt may leave an empty/partial file behind.
            try:
                bag_path.unlink()
            except FileNotFoundError:
                pass
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if bag_path is not None:
            config.enable_record_to_file(str(bag_path))
        try:
            active = pipeline.start(config)
            print(f"Using {width}x{height}@{fps} for Depth(Z16) + Color(BGR8)", flush=True)
            return pipeline, active, (width, height, fps)
        except Exception as exc:  # a profile may not be supported by a particular firmware/device
            errors.append(f"{width}x{height}@{fps}: {exc}")
            try:
                pipeline.stop()
            except Exception:
                pass
    raise RuntimeError("Could not start any requested shared RGB/depth profile:\n" + "\n".join(errors))


def main() -> int:
    args = parse_args()
    if args.frames < 0 or args.seconds < 0:
        raise SystemExit("--frames and --seconds must be >= 0")
    if args.preview_near < 0 or args.preview_far <= args.preview_near:
        raise SystemExit("Depth preview range must satisfy 0 <= near < far")

    rs = load_realsense()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output or Path.cwd() / f"realsense_capture_{stamp}").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    folders = mkdirs(out)
    bag_path = out / "recording.bag" if args.record_bag else None

    pipeline = None
    preview_created = False
    records: list[dict[str, Any]] = []
    depth_scale = None
    try:
        pipeline, active, selected_profile = start_pipeline(rs, args.serial, args.profile, bag_path)
        device = active.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = active.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = active.get_stream(rs.stream.depth).as_video_stream_profile()
        color_intr = color_profile.get_intrinsics()
        depth_intr = depth_profile.get_intrinsics()
        depth_to_color = depth_profile.get_extrinsics_to(color_profile)

        # SDK processing block: this is the actual calibrated depth -> color reprojection.
        align = rs.align(rs.stream.color)

        metadata: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "device": {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
                "product_line": device.get_info(rs.camera_info.product_line),
                "librealsense_version": getattr(rs, "__version__", "unknown"),
            },
            "selected_profile": {"width": selected_profile[0], "height": selected_profile[1], "fps": selected_profile[2]},
            "streams": {"depth": profile_dict(active.get_stream(rs.stream.depth)),
                        "color": profile_dict(active.get_stream(rs.stream.color))},
            "depth_scale_m_per_unit": depth_scale,
            "depth_to_color_extrinsics": {
                # RealSense SDK supplies the depth-camera -> color-camera transform.
                "rotation_column_major": [float(x) for x in depth_to_color.rotation],
                "translation_m": [float(x) for x in depth_to_color.translation],
            },
            "alignment": "pyrealsense2.align(rs.stream.color); aligned depth shares color pixel coordinates/resolution",
            "saved_formats": {
                "rgb": "BGR8 PNG",
                "depth": "float32 NPY; metres; aligned to color pixel grid; directly compatible with current WAPR RGB-D runner",
                "depth_aligned_mm": "uint16 PNG; millimetres; aligned to color pixel grid",
                "depth_raw_z16": "uint16 PNG; original sensor Z16 units; native depth-camera pixel grid",
                "depth_preview": "8-bit grayscale and Turbo color previews; visualization only, not metric depth",
                "recording.bag": "optional raw device streams; reapply SDK alignment during playback",
            },
            "rgb_camera_K_file": "cam_K.txt",
            "max_sync_ms_warning_threshold": float(args.max_sync_ms),
            "frames": [],
        }
        np.savetxt(out / "cam_K.txt", np.asarray([[color_intr.fx, 0, color_intr.ppx],
                                                    [0, color_intr.fy, color_intr.ppy],
                                                    [0, 0, 1]], dtype=np.float64), fmt="%.10g")
        (out / "camera_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        print(f"Device: {metadata['device']['name']} serial={metadata['device']['serial']}")
        print(f"Depth scale: {depth_scale:.9f} m/unit")
        print("Press q or Esc in the preview window to stop. Ctrl-C also works.")

        started = time.monotonic()
        frame_index = 0
        last_report = started
        sync_deltas: list[float] = []
        while True:
            frames = pipeline.wait_for_frames(5000)
            if not frames:
                continue
            raw_depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not raw_depth_frame or not color_frame:
                continue

            aligned = align.process(frames)
            aligned_depth_frame = aligned.get_depth_frame()
            aligned_color_frame = aligned.get_color_frame() if hasattr(aligned, "get_color_frame") else aligned.first(rs.stream.color)
            if not aligned_depth_frame or not aligned_color_frame:
                continue

            color = np.asanyarray(aligned_color_frame.get_data())  # BGR8
            raw_depth_u16 = np.asanyarray(raw_depth_frame.get_data()).copy()
            aligned_depth_u16 = np.asanyarray(aligned_depth_frame.get_data()).copy()
            if "aligned_depth_intrinsics" not in metadata:
                metadata["aligned_depth_intrinsics"] = intrinsics_dict(
                    aligned_depth_frame.profile.as_video_stream_profile().get_intrinsics())
            aligned_depth_m = aligned_depth_u16.astype(np.float32) * depth_scale
            aligned_depth_mm = np.clip(np.rint(aligned_depth_m * 1000.0), 0, 65535).astype(np.uint16)

            if color.shape[:2] != aligned_depth_u16.shape[:2]:
                raise RuntimeError(f"SDK alignment returned different shapes: RGB={color.shape[:2]} depth={aligned_depth_u16.shape[:2]}")

            frame_index += 1
            frame_id = f"{frame_index - 1:06d}"
            rgb_ts = float(color_frame.get_timestamp())
            depth_ts = float(raw_depth_frame.get_timestamp())
            delta_ms = abs(rgb_ts - depth_ts)
            sync_deltas.append(delta_ms)

            # Persist per-frame sensor data.
            if not cv2.imwrite(str(folders["rgb"] / f"{frame_id}.png"), color):
                raise OSError(f"Failed to write RGB frame {frame_id}")
            if not cv2.imwrite(str(folders["depth_aligned_mm"] / f"{frame_id}.png"), aligned_depth_mm):
                raise OSError(f"Failed to write aligned depth frame {frame_id}")
            if not cv2.imwrite(str(folders["depth_raw_z16"] / f"{frame_id}.png"), raw_depth_u16):
                raise OSError(f"Failed to write raw depth frame {frame_id}")
            # The WAPR batch runner consumes depth/<frame>.npy in metres.
            np.save(folders["depth"] / f"{frame_id}.npy", aligned_depth_m, allow_pickle=False)

            gray, depth_color = depth_preview(aligned_depth_m, args.preview_near, args.preview_far)
            cv2.imwrite(str(folders["depth_preview"] / f"{frame_id}_gray8.png"), gray)
            cv2.imwrite(str(folders["depth_preview"] / f"{frame_id}_color.png"), depth_color)

            rec = {
                "frame": frame_id,
                "rgb_timestamp_ms": rgb_ts,
                "depth_timestamp_ms": depth_ts,
                "sync_delta_ms": delta_ms,
                "timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
                "rgb_frame_number": int(color_frame.get_frame_number()),
                "depth_frame_number": int(raw_depth_frame.get_frame_number()),
                "rgb_file": f"rgb/{frame_id}.png",
                "aligned_depth_mm_file": f"depth_aligned_mm/{frame_id}.png",
                "aligned_depth_m_file": f"depth/{frame_id}.npy",
                "raw_depth_z16_file": f"depth_raw_z16/{frame_id}.png",
                "aligned_depth_valid_fraction": float(np.count_nonzero(aligned_depth_u16) / aligned_depth_u16.size),
            }
            records.append(rec)

            metadata["frames"].append(rec)
            # Update metadata every 30 frames so interrupted captures still have useful information.
            if frame_index % 30 == 0:
                (out / "camera_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                write_conversion_json(out, metadata, records, args.max_sync_ms, bag_path)

            if not args.no_preview:
                # Side-by-side RGB / aligned depth and RGB blended with colorized depth.
                overlay = color.copy()
                valid = aligned_depth_u16 > 0
                overlay[valid] = cv2.addWeighted(color[valid], 0.45, depth_color[valid], 0.55, 0)
                preview = np.hstack((color, depth_color, overlay))
                if preview.shape[1] > 1800:
                    scale = 1800.0 / preview.shape[1]
                    preview = cv2.resize(preview, (1800, int(preview.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                cv2.putText(preview, f"{frame_id}  sync={delta_ms:.2f} ms  valid={rec['aligned_depth_valid_fraction']:.1%}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("RealSense | RGB | aligned depth | overlay", preview)
                preview_created = True
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

            now = time.monotonic()
            if now - last_report >= 2.0:
                recent = sync_deltas[-60:]
                print(f"frames={frame_index} sync_median={np.median(recent):.2f}ms "
                      f"sync_max={np.max(recent):.2f}ms "
                      f"aligned_valid={rec['aligned_depth_valid_fraction']:.1%}", flush=True)
                last_report = now
            if args.frames and frame_index >= args.frames:
                break
            if args.seconds and now - started >= args.seconds:
                break

        elapsed = max(time.monotonic() - started, 1e-6)
        metadata["capture_summary"] = {
            "frames": frame_index,
            "elapsed_seconds": elapsed,
            "effective_fps": frame_index / elapsed,
            "sync_delta_median_ms": float(np.median(sync_deltas)) if sync_deltas else None,
            "sync_delta_p95_ms": float(np.percentile(sync_deltas, 95)) if sync_deltas else None,
            "sync_delta_max_ms": float(np.max(sync_deltas)) if sync_deltas else None,
            "frames_over_sync_threshold": int(sum(x > args.max_sync_ms for x in sync_deltas)),
            "aligned_depth_valid_fraction_mean": float(np.mean([x["aligned_depth_valid_fraction"] for x in records])) if records else None,
            "recording_bag": str(bag_path) if bag_path else None,
        }
        (out / "camera_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        write_conversion_json(out, metadata, records, args.max_sync_ms, bag_path)
        print("\nCapture complete:")
        print(f"  output: {out}")
        print(f"  frames: {frame_index}")
        print(f"  WAPR-ready aligned depth: {folders['depth']} (metres, .npy)")
        print(f"  RGB intrinsics: {out / 'cam_K.txt'}")
        if bag_path:
            print(f"  RealSense bag: {bag_path}")
        return 0

    except KeyboardInterrupt:
        print("\nCapture interrupted; finalizing metadata.")
        if records:
            metadata["frames"] = records
            metadata["capture_summary"] = {"frames": len(records), "interrupted": True}
            (out / "camera_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            write_conversion_json(out, metadata, records, args.max_sync_ms, bag_path)
        print(f"Output: {out}")
        return 130
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if preview_created:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
