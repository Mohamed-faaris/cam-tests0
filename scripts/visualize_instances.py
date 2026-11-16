#!/usr/bin/env python3
"""Render YOLO instance masks with distinct colors and stable per-image IDs."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# BGR palette chosen for separation on typical RGB camera scenes.
PALETTE = [
    (255, 90, 65), (30, 190, 255), (80, 220, 90), (220, 80, 245),
    (20, 220, 220), (190, 130, 255), (255, 175, 35), (90, 145, 255),
    (155, 225, 55), (255, 115, 175), (90, 210, 165), (210, 170, 75),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input image")
    parser.add_argument("--weights", default="yolo26s-seg.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=Path("outputs/instances.jpg"))
    args = parser.parse_args()

    image = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.source}")

    model = YOLO(args.weights)
    result = model.predict(str(args.source), imgsz=args.imgsz, conf=args.conf,
                           iou=0.50, verbose=False)[0]
    overlay = image.copy()
    instances = []

    if result.masks is not None and result.boxes is not None:
        for idx, (box, polygon) in enumerate(zip(result.boxes, result.masks.xy), 1):
            if len(polygon) < 3:
                continue
            color = PALETTE[(idx - 1) % len(PALETTE)]
            points = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [points], color, lineType=cv2.LINE_AA)
            edge = tuple(int(channel * 0.78) for channel in color)
            cv2.polylines(overlay, [points], True, edge, 2, cv2.LINE_AA)
            moments = cv2.moments(points)
            if moments["m00"]:
                center = (int(moments["m10"] / moments["m00"]),
                          int(moments["m01"] / moments["m00"]))
            else:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
            instances.append({
                "id": idx,
                "class": model.names[int(box.cls.item())],
                "confidence": float(box.conf.item()),
                "center": center,
                "color": color,
            })

    # Tint only segmented pixels; preserve the unsegmented scene.
    rendered = cv2.addWeighted(image, 0.48, overlay, 0.52, 0)

    # Small ID discs on the objects. Nudge nearby discs apart to prevent collisions.
    placed = []
    for item in instances:
        x, y = item["center"]
        for _ in range(12):
            if all((x-px)**2 + (y-py)**2 >= 42**2 for px, py in placed):
                break
            y += 18
        x = min(max(x, 20), rendered.shape[1] - 20)
        y = min(max(y, 20), rendered.shape[0] - 20)
        placed.append((x, y))
        cv2.circle(rendered, (x, y), 17, item["color"], -1, cv2.LINE_AA)
        cv2.circle(rendered, (x, y), 17, (25, 25, 25), 2, cv2.LINE_AA)
        label = f"{item['id']:02d}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
        cv2.putText(rendered, label, (x - tw // 2, y + th // 2 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 2, cv2.LINE_AA)

    # Compact legend on the clear left side; no object bounding boxes are drawn.
    if instances:
        x0, y0 = 18, 18
        row_h, panel_w = 29, 252
        panel_h = 42 + row_h * len(instances)
        panel = rendered.copy()
        cv2.rectangle(panel, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 24, 30), -1)
        rendered = cv2.addWeighted(panel, 0.78, rendered, 0.22, 0)
        cv2.putText(rendered, "INSTANCE MASKS", (x0 + 12, y0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)
        for row, item in enumerate(instances):
            cy = y0 + 49 + row * row_h
            cv2.circle(rendered, (x0 + 20, cy - 5), 7, item["color"], -1, cv2.LINE_AA)
            text = f"ID {item['id']:02d}  {item['class']}  {item['confidence']:.2f}"
            cv2.putText(rendered, text, (x0 + 36, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (245, 245, 245), 1, cv2.LINE_AA)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise SystemExit(f"Could not write output: {args.output}")
    print(f"Instances: {len(instances)}")
    for item in instances:
        print(f"ID {item['id']:02d}: {item['class']} ({item['confidence']:.3f})")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
