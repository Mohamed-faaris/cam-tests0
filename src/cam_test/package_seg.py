"""Train and run Ultralytics instance segmentation on the Package-Seg dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "package-seg.yaml"
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "runs" / "segment" / "package-seg" / "weights" / "best.pt"
)


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: uv sync"
        ) from exc
    return YOLO


def _custom_model(weights: Path):
    if not weights.is_file():
        raise SystemExit(
            f"Trained segmentation weights not found: {weights}\n"
            "Train first with: uv run python -m cam_test.package_seg train"
        )
    return _load_yolo()(str(weights))


def train(args: argparse.Namespace) -> None:
    YOLO = _load_yolo()
    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "segment"),
        name="package-seg",
        exist_ok=True,
    )


def validate(args: argparse.Namespace) -> None:
    model = _custom_model(args.weights)
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(metrics)


def predict(args: argparse.Namespace) -> None:
    model = _custom_model(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        show=args.show,
        project=str(PROJECT_ROOT / "runs" / "predict"),
        name="package-seg",
        exist_ok=True,
    )


def live(args: argparse.Namespace) -> None:
    import cv2

    model = _custom_model(args.weights)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Package Instance Segmentation"
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame read failed.", file=sys.stderr)
                break

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            annotated = result.plot()
            cv2.imshow(window_name, annotated)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run package instance segmentation."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train on Package-Seg.")
    train_parser.add_argument("--model", default="yolo26n-seg.pt")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--device", default=None)
    train_parser.set_defaults(func=train)

    val_parser = commands.add_parser("val", help="Evaluate trained weights.")
    val_parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    val_parser.add_argument("--imgsz", type=int, default=640)
    val_parser.add_argument("--batch", type=int, default=8)
    val_parser.add_argument("--device", default=None)
    val_parser.set_defaults(func=validate)

    predict_parser = commands.add_parser(
        "predict", help="Segment an image or video file."
    )
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    predict_parser.add_argument("--imgsz", type=int, default=640)
    predict_parser.add_argument("--conf", type=float, default=0.25)
    predict_parser.add_argument("--device", default=None)
    predict_parser.add_argument("--show", action="store_true")
    predict_parser.set_defaults(func=predict)

    live_parser = commands.add_parser(
        "live", help="Run segmentation on a webcam."
    )
    live_parser.add_argument("--camera", type=int, default=0)
    live_parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    live_parser.add_argument("--width", type=int, default=1280)
    live_parser.add_argument("--height", type=int, default=720)
    live_parser.add_argument("--imgsz", type=int, default=640)
    live_parser.add_argument("--conf", type=float, default=0.25)
    live_parser.add_argument("--device", default=None)
    live_parser.set_defaults(func=live)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
