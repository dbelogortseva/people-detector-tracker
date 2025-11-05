import numpy as np
import cv2


class KalmanFilter:
    """
    Реализация фильтра Калмана для трекинга bbox + для сглаживания перемещения .
    Состояние: [c_x, c_y, w, h, dc_x/dt, dc_y/dt, dw/dt, dh/dt]

    Для каждого трека хранится:
        - x, P     : состояние фильтра Калмана
        - age, miss: возраст и число пропусков
        - conf     : последняя уверенность детектора (из модели YOLO)
    """

    def __init__(self, fps, init_bboxes=None, iou_thr=0.1, max_age=2, min_area=1.0):
        self.dt = 1.0 / max(1e-6, float(fps))
        self.I = np.eye(8, dtype=float)

        self.F = np.eye(8, dtype=float)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = self.dt

        self.H = np.zeros((4, 8), dtype=float)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        q = 1e-2
        self.Q = np.diag([q] * 8)
        self.R = np.diag([1.0] * 4)

        self.iou_thr = float(iou_thr)
        self.max_age = int(max_age)
        self.min_area = float(min_area)

        # tid -> {"x","P","age","miss","conf"}
        self.tracks = {}
        self.next_id = 0

        for item in init_bboxes:
            bbox = item.get("bbox")
            conf = item.get("conf")
            if bbox:
                self.create_track_from_bbox(bbox, conf=conf)

    def create_track_from_bbox(self, bbox, conf=None):
        """Создаёт новый трек из bbox и начального conf"""
        
        x1, y1, x2, y2 = map(float, bbox)
        cx, cy, w, h = self.xyxy_to_cxcywh(x1, y1, x2, y2)
        x = np.zeros((8, 1), dtype=float)
        x[0, 0], x[1, 0], x[2, 0], x[3, 0] = cx, cy, w, h
        P = np.eye(8, dtype=float) * 10.0
        self.tracks[self.next_id] = {"x": x, "P": P, "age": 0, "miss": 0, "conf": conf}
        self.next_id += 1

    def predict_bboxes(self):
        """Прогноз перемещения bbox для всех треков"""
        
        out = []
        for tid, t in self.tracks.items():
            x, P = t["x"], t["P"]
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            t["x"], t["P"] = x, P
            t["age"] += 1
            out.append((tid, self.cxcywh_to_xyxy(*x[:4, 0])))
        return out

    def update_bboxes_with_model(self, new_bboxes=None, new_confs=None, alpha=0.3):
        """
        Обновление согласно предсказанию + на основе модели.

        Параметры:
            new_bboxes: новые bbox, посчитанные YOLO
            new_confs: новые conf из инференса YOLO
            alpha (от 0 до 1): смешивание измерения и предсказания по центрам/размерам:
                  z = alpha * H @ x_pred + (1-alpha) * z

        Возвращает:
            список (tid, [x1,y1,x2,y2]) для всех текущих треков.
        """
        dets_xyxy = []
        dets_conf = []

        
        # Выбираем bbox, которые имеют не слишком маленькую площадь
        if new_bboxes is not None:
            if len(new_bboxes) > 0 and isinstance(new_bboxes[0], dict):
                for d in new_bboxes:
                    if "bbox" not in d:
                        continue
                    x1, y1, x2, y2 = map(float, d["bbox"])
                    if (x2 > x1) and (y2 > y1) and ((x2 - x1) * (y2 - y1) >= self.min_area):
                        dets_xyxy.append([x1, y1, x2, y2])
                        dets_conf.append(float(d.get("conf")) if d.get("conf") is not None else None)
        

        dets_xyxy = np.array(dets_xyxy, dtype=float) if dets_xyxy else np.zeros((0, 4), dtype=float)

        # Считаем предсказания от фильтра Калмана, которые затем совместим с предсказаниями от модели YOLO
        tids = list(self.tracks.keys())
        preds_xyxy = []
        for tid in tids:
            t = self.tracks[tid]
            x, P = t["x"], t["P"]
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q
            t["_x_pred"] = x_pred
            t["_P_pred"] = P_pred
            preds_xyxy.append(self.cxcywh_to_xyxy(*x_pred[:4, 0]))
        preds_xyxy = np.array(preds_xyxy, dtype=float) if preds_xyxy else np.zeros((0, 4), dtype=float)

        # Жадным алгоритмом находим оптимальные пары между инференсом модели и предсказанием фильтра
        # Оцениваем по метрике: площадь пересечения/площадь объединения
        matches, unmatched_trk, unmatched_det = self.match_greedy(preds_xyxy, dets_xyxy, self.iou_thr)

        # Если для bbox была найдена пара из предикта, то итоговым результатом будет сглаженная оценка YOLO и фильтра Калмана
        for mi in matches:
            ti, di = mi
            tid = tids[ti]
            z_cx, z_cy, z_w, z_h = self.xyxy_to_cxcywh(*dets_xyxy[di])
            z = np.array([[z_cx], [z_cy], [z_w], [z_h]], dtype=float)

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

            # Меняем последнее значение conf на новое значение, полученное от YOLO
            if dets_conf and di < len(dets_conf):
                self.tracks[tid]["conf"] = dets_conf[di]

        # Если есть bbox, у которых нет нового значения, то двигаем их с помощью предсказания от фильтра Калмана
        for ti in unmatched_trk:
            tid = tids[ti]
            self.tracks[tid]["x"] = self.tracks[tid]["_x_pred"]
            self.tracks[tid]["P"] = self.tracks[tid]["_P_pred"]
            self.tracks[tid]["miss"] += 1
            self.tracks[tid]["age"] += 1
            # conf оставляем прежним (последнее известное значение)

        # Если есть треки, которых раньше не было, то создаем новые треки
        for di in unmatched_det:
            conf_val = None
            if dets_conf and di < len(dets_conf):
                conf_val = dets_conf[di]
            self.create_track_from_bbox(dets_xyxy[di], conf=conf_val)

        # Если трек уже неактуальный, то удаляем его
        to_del = [tid for tid, t in self.tracks.items() if t["miss"] > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        out = []
        for tid, t in self.tracks.items():
            out.append((tid, self.cxcywh_to_xyxy(*t["x"][:4, 0])))
        return out

    def match_greedy(self, preds_xyxy, dets_xyxy, iou_thr):
        """
        Жадное сопоставление предсказанных и детектированных боксов по iou.

        Находит пары (трек, детекция) с максимальным iou выше порога iou_thr.
        Возвращает списки:
            matches — совпавшие пары индексов,
            unmatched_trk — треки без совпадений,
            unmatched_det — детекции без совпадений.
        """
        
        preds_xyxy = np.array(preds_xyxy, dtype=float)
        dets_xyxy = np.array(dets_xyxy, dtype=float)

        if preds_xyxy.shape[0] == 0 and dets_xyxy.shape[0] == 0:
            return [], [], []
        if preds_xyxy.shape[0] == 0:
            return [], [], list(range(dets_xyxy.shape[0]))
        if dets_xyxy.shape[0] == 0:
            return [], list(range(preds_xyxy.shape[0])), []

        iou_mat = np.zeros((preds_xyxy.shape[0], dets_xyxy.shape[0]), dtype=float)
        for i in range(iou_mat.shape[0]):
            for j in range(iou_mat.shape[1]):
                iou_mat[i, j] = self.iou(preds_xyxy[i], dets_xyxy[j])

        matches, used_pred, used_det = [], set(), set()
        flat = [(iou_mat[i, j], i, j)
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
        """
        Преобразует формат bbox из (x1, y1, x2, y2) в (cx, cy, w, h).

        Возвращает:
            cx, cy — координаты центра бокса,
            w, h — ширину и высоту бокса.
        """
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cx, cy, w, h

    @staticmethod
    def cxcywh_to_xyxy(cx, cy, w, h):
        """
        Преобразует формат bbox из (cx, cy, w, h) в (x1, y1, x2, y2).

        Возвращает:
            x1, y1 — координаты левого верхнего угла,
            x2, y2 — координаты правого нижнего угла бокса.
        """
        return float(cx - w / 2.0), float(cy - h / 2.0), float(cx + w / 2.0), float(cy + h / 2.0)

    @staticmethod
    def iou(a, b):
        """
        Вычисляет пересечение по IoU (Intersection over Union) между двумя боксами.

        Аргументы:
            a, b — боксы в формате (x1, y1, x2, y2).

        Возвращает:
            Значение IoU в диапазоне [0, 1].
        """
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
        return inter / ua

    def draw_bboxes(self, frame, line_width=2):
        vis = frame.copy()
        color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for tid, t in self.tracks.items():
            # Рисуем прямоугольник
            x1, y1, x2, y2 = self.cxcywh_to_xyxy(*t["x"][:4, 0])
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_width)

            # Текст: три строки — id, Person, conf
            y0 = max(20, y1 - 10)
            cv2.putText(vis, f"id:{tid}", (x1, y0), font, 0.7, color, 2, cv2.LINE_AA)
            cv2.putText(vis, "Person", (x1, y0 + 20), font, 0.7, color, 2, cv2.LINE_AA)

            conf = t.get("conf")
            if conf is not None:
                cv2.putText(vis, f"{float(conf):.2f}", (x1, y0 + 40), font, 0.7, color, 2, cv2.LINE_AA)

        return vis
