"""Module kalman_tracker — part of YOLO+Kalman tracking pipeline."""

import numpy as np
import cv2

class KalmanFilter:
    """
    Простая реализация Kalman Filter для трекинга bbox:
    состояние: [c_x, c_y, w, h, dcx/dt, dcy/dt, dw/dt, dh/dt]
    """
    def __init__(self, fps, init_bboxes=None, iou_thr=0.1, max_age=2, min_area=1.0):
        self.dt = 1.0 / max(1e-6, float(fps))
        self.I = np.eye(8, dtype=float)

        self.F = np.eye(8, dtype=float)
        self.F[0,4]=self.F[1,5]=self.F[2,6]=self.F[3,7]=self.dt

        self.H = np.zeros((4,8), dtype=float)
        self.H[0,0]=self.H[1,1]=self.H[2,2]=self.H[3,3]=1.0

        q = 1e-2
        self.Q = np.diag([q]*8)
        self.R = np.diag([1.0]*4)

        self.iou_thr = float(iou_thr)
        self.max_age = int(max_age)
        self.min_area = float(min_area)

        self.tracks = {}
        self.next_id = 0

        if init_bboxes:
            for bbox in init_bboxes:
                self.create_track_from_bbox(bbox)

    def create_track_from_bbox(self, bbox):
        x1, y1, x2, y2 = map(float, bbox)
        cx, cy, w, h = self.xyxy_to_cxcywh(x1, y1, x2, y2)
        x = np.zeros((8,1), dtype=float)
        x[0,0], x[1,0], x[2,0], x[3,0] = cx, cy, w, h
        P = np.eye(8, dtype=float) * 10.0
        self.tracks[self.next_id] = {"x":x, "P":P, "age":0, "miss":0}
        self.next_id += 1

    def predict_bboxes(self):
        out = []
        for tid, t in self.tracks.items():
            x, P = t["x"], t["P"]
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            t["x"], t["P"] = x, P
            t["age"] += 1
            out.append((tid, self.cxcywh_to_xyxy(*x[:4,0])))
        return out

    def update_bboxes_with_model(self, new_bboxes=None, alpha=0.3):
        dets_xyxy = []
        if new_bboxes is not None:
            for b in new_bboxes:
                x1, y1, x2, y2 = map(float, b)
                if (x2 > x1) and (y2 > y1) and ((x2-x1)*(y2-y1) >= self.min_area):
                    dets_xyxy.append([x1, y1, x2, y2])
        dets_xyxy = np.array(dets_xyxy, dtype=float) if dets_xyxy else np.zeros((0,4), dtype=float)

        tids = list(self.tracks.keys())
        preds_xyxy = []
        for tid in tids:
            t = self.tracks[tid]
            x, P = t["x"], t["P"]
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q
            t["_x_pred"] = x_pred
            t["_P_pred"] = P_pred
            preds_xyxy.append(self.cxcywh_to_xyxy(*x_pred[:4,0]))
        preds_xyxy = np.array(preds_xyxy, dtype=float) if preds_xyxy else np.zeros((0,4), dtype=float)

        matches, unmatched_trk, unmatched_det = self.match_greedy(preds_xyxy, dets_xyxy, self.iou_thr)

        for mi in matches:
            ti, di = mi
            tid = tids[ti]
            z_cx, z_cy, z_w, z_h = self.xyxy_to_cxcywh(*dets_xyxy[di])
            z = np.array([[z_cx],[z_cy],[z_w],[z_h]], dtype=float)

            x_pred = self.tracks[tid]["_x_pred"]
            P_pred = self.tracks[tid]["_P_pred"]

            if alpha > 0.0:
                z = alpha * (self.H @ x_pred) + (1.0 - alpha) * z

            y = z - (self.H @ x_pred)
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            x_new = x_pred + K @ y
            P_new = (self.I - K @ self.H) @ P_pred

            self.tracks[tid]["x"] = x_new
            self.tracks[tid]["P"] = P_new
            self.tracks[tid]["miss"] = 0

        for ti in unmatched_trk:
            tid = tids[ti]
            self.tracks[tid]["x"] = self.tracks[tid]["_x_pred"]
            self.tracks[tid]["P"] = self.tracks[tid]["_P_pred"]
            self.tracks[tid]["miss"] += 1
            self.tracks[tid]["age"] += 1

        for di in unmatched_det:
            self.create_track_from_bbox(dets_xyxy[di])

        to_del = [tid for tid,t in self.tracks.items() if t["miss"] > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        out = []
        for tid, t in self.tracks.items():
            out.append((tid, self.cxcywh_to_xyxy(*t["x"][:4,0])))
        return out

    def match_greedy(self, preds_xyxy, dets_xyxy, iou_thr):
        preds_xyxy = np.array(preds_xyxy, dtype=float)
        dets_xyxy = np.array(dets_xyxy, dtype=float)

        if preds_xyxy.shape[0]==0 and dets_xyxy.shape[0]==0:
            return [], [], []
        if preds_xyxy.shape[0]==0:
            return [], [], list(range(dets_xyxy.shape[0]))
        if dets_xyxy.shape[0]==0:
            return [], list(range(preds_xyxy.shape[0])), []

        iou_mat = np.zeros((preds_xyxy.shape[0], dets_xyxy.shape[0]), dtype=float)
        for i in range(iou_mat.shape[0]):
            for j in range(iou_mat.shape[1]):
                iou_mat[i,j] = self.iou(preds_xyxy[i], dets_xyxy[j])

        matches, used_pred, used_det = [], set(), set()
        flat = [(iou_mat[i,j], i, j)
                for i in range(iou_mat.shape[0])
                for j in range(iou_mat.shape[1])]
        flat.sort(reverse=True)
        for iou, i, j in flat:
            if i in used_pred or j in used_det:
                continue
            if iou >= iou_thr:
                matches.append((i, j))
                used_pred.add(i)
                used_det.add(j)

        unmatched_trk = [i for i in range(preds_xyxy.shape[0]) if i not in used_pred]
        unmatched_det = [j for j in range(dets_xyxy.shape[0]) if j not in used_det]
        return matches, unmatched_trk, unmatched_det

    @staticmethod
    def xyxy_to_cxcywh(x1, y1, x2, y2):
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        cx = (x1 + x2)/2.0
        cy = (y1 + y2)/2.0
        return cx, cy, w, h

    @staticmethod
    def cxcywh_to_xyxy(cx, cy, w, h):
        return float(cx - w/2.0), float(cy - h/2.0), float(cx + w/2.0), float(cy + h/2.0)

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2,bx2), min(ay2,by2)
        iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
        inter = iw*ih
        ua = max(1e-6, (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
        return inter/ua

    def draw_bboxes(self, frame, labels=True, boxes=True, line_width=2):
        vis = frame.copy()
        if boxes:
            for tid, t in self.tracks.items():
                x1,y1,x2,y2 = self.cxcywh_to_xyxy(*t["x"][:4,0])
                x1=int(round(x1)); y1=int(round(y1)); x2=int(round(x2)); y2=int(round(y2))
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), line_width)
                if labels:
                    cv2.putText(vis, f"id:{tid}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        return vis
