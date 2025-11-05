"""Core logic for YOLOv8 detection combined with Kalman tracking."""

import cv2
from tqdm import trange
from ultralytics import YOLO
from src.kalman_tracker import KalmanFilter


def run_with_kalman(cap, out, model, conf, class_id, frame_count,
    fps, w, h, stride, iou_thr, max_age):
    """Run YOLOv8 + Kalman tracking on a video stream.

    Args:
        cap (cv2.VideoCapture): Video capture object.
        out (cv2.VideoWriter): Output video writer.
        model (YOLO): Loaded YOLOv8 model.
        fps (float): Frames per second of the input video.
        frame_count (int): Total number of frames.
    """
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame from the video.")

    res = model(frame, conf=conf, classes=class_id, verbose=False)[0]
    init_boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []

    # Initialize Kalman filter with initial detections
    kf = KalmanFilter(fps=fps, init_bboxes=init_boxes)

    # Process the remaining frames
    for i in trange(frame_count-1, desc="Kalman+YOLO"):
        flag, frame = cap.read()
        if not flag:
            break

        kf.predict_bboxes()

        if i % stride == 0:
            res = model(frame, conf=conf, classes=class_id, verbose=False)[0]
            new_boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []
            _ = kf.update_bboxes_with_model(new_bboxes=new_boxes, alpha=0.1)

        vis = kf.draw_bboxes(frame, labels=True, boxes=True, line_width=3)
        out.write(vis)


def run_yolo_only(cap, out, model, conf, class_id, frame_count):
    """Run YOLOv8-only detection on video frames.

    Args:
        cap (cv2.VideoCapture): Video capture object.
        out (cv2.VideoWriter): Output video writer.
        model (YOLO): Loaded YOLOv8 model.
        frame_count (int): Total frame count.
        conf (float): Confidence threshold for YOLO.
    """
    for _ in trange(frame_count-1, desc="YOLO"):
        ok, frame = cap.read()
        if not ok:
            break

        res = model(frame, conf=conf, classes=[0], verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes is not None else []

        for b in boxes:
            x1, y1, x2, y2 = map(int, b[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out.write(frame)
