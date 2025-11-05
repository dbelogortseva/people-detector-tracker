"""
Примеры использования:
    python main.py --input crowd.mp4 --mode yolo
    python main.py --input crowd.mp4 --mode kalman --stride 10
"""

import argparse
import cv2
from ultralytics import YOLO
from pathlib import Path
from src.detect_and_track import run_yolo_only, run_with_kalman


def main():
    """Command-line interface for YOLOv8 + Kalman tracking."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference with optional Kalman tracking."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input video.")
    parser.add_argument("--output", type=str, default="outputs/detected.mp4",
                        help="Path to output video.")
    parser.add_argument("--model", type=str, default="weights/yolov8n.pt",
                        help="YOLO model weights.")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="YOLO confidence threshold (0–1).")
    parser.add_argument("--mode", type=str, choices=["yolo", "kalman"], default="kalman",
                        help="Mode: 'yolo' = pure YOLO; 'kalman' = YOLO + Kalman filter.")
    parser.add_argument("--stride", type=int, default=10,
                        help="(kalman) Run detector every N frames.")
    parser.add_argument("--iou-thr", type=float, default=0.3,
                        help="(kalman) IoU threshold for matching.")
    parser.add_argument("--max-age", type=int, default=30,
                        help="(kalman) Max missed frames before deleting a track.")

    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"No such input file: {inp}")

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {inp}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    model = YOLO(args.model)

    try:
        if args.mode == "yolo":
            run_yolo_only(cap, out, model, args.conf, frame_count)
        else:
            run_with_kalman(
                cap, out, model, args.conf,
                frame_count, fps, w, h, args.stride, args.iou_thr, args.max_age
            )
    finally:
        cap.release()
        out.release()

    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
