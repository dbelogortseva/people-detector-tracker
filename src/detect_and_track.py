import cv2
from tqdm import trange
from ultralytics import YOLO
from src.kalman_tracker import KalmanFilter


def run_with_kalman(cap, out, model, conf, frame_count, fps, w, h, stride, iou_thr, max_age):
    """ Запуск модели YOLO + фильтрация с помощью фильтра Калмана => на выходе детекция + стабильный трекинг

    Аргументы:
        cap (cv2.VideoCapture):
            Открытый видеопоток (входное видео).
        out (cv2.VideoWriter):
            Объект для записи обработанных кадров в выходное видео.
        model (ultralytics.YOLO):
            Загруженная модель YOLOv8 для детекции объектов.
        conf (float):
            Порог уверенности (0–1) для фильтрации детекций YOLO.
        frame_count (int):
            Общее количество кадров во входном видео.
        fps (float):
            Частота кадров исходного видео.
        w (int):
            Ширина кадра (в пикселях).
        h (int):
            Высота кадра (в пикселях).
        stride (int):
            Интервал обновления YOLO-детектора.
            Пример: stride=10 → YOLO вызывается каждые 10 кадров,
            между вызовами работает только фильтр Калмана.
        iou_thr (float):
            Порог IoU для сопоставления детекций с текущими треками.
        max_age (int):
            Максимальное количество кадров, в течение которых
            трек может "теряться" без обновления от YOLO,
            прежде чем будет удалён.

    """
    # Считываем первый кадр
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame from the video.")

    res = model(frame, conf=conf, classes=[0], verbose=False)[0]
    init_dets = []
    if res.boxes is not None:
        for b in res.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            c = float(b.conf[0]) if getattr(b, "conf", None) is not None else None
            cls_id = int(b.cls[0]) if getattr(b, "cls", None) is not None else None
            init_dets.append({"bbox": [x1, y1, x2, y2], "conf": c, "cls": cls_id})

    # Загружаем начальные боксы в класс для фильтра Калмана
    kf = KalmanFilter(fps=fps, init_bboxes=init_dets)

    # Считываем все последующие кадры
    for i in trange(frame_count-1, desc="YOLO+Kalman"):
        
        flag, frame = cap.read()
        if not flag:
            break

        kf.predict_bboxes()

        if i % stride == 0:
            res = model(frame, conf=conf, classes=[0], verbose=False)[0]
            new_dets = []
            if res.boxes is not None:
                for b in res.boxes:
                    x1, y1, x2, y2 = map(float, b.xyxy[0])
                    c = float(b.conf[0]) if getattr(b, "conf", None) is not None else None
                    cls_id = int(b.cls[0]) if getattr(b, "cls", None) is not None else None
                    new_dets.append({"bbox": [x1, y1, x2, y2], "conf": c, "cls": cls_id})

                kf.update_bboxes_with_model(new_bboxes=new_dets, alpha=0.1)

        vis = kf.draw_bboxes(frame, line_width=3)
        out.write(vis)


def run_yolo_only(cap, out, model, conf, frame_count):
    """ Запуск модели YOLO (модель применяется для каждого фрейма)

    Аргументы:
        cap (cv2.VideoCapture):
            Открытый видеопоток (входное видео).
        out (cv2.VideoWriter):
            Объект для записи обработанных кадров в выходное видео.
        model (ultralytics.YOLO):
            Загруженная модель YOLOv8 для детекции объектов.
        conf (float):
            Порог уверенности (0–1) для фильтрации детекций YOLO.
        frame_count (int):
            Общее количество кадров во входном видео.

    """
    for _ in trange(frame_count-1, desc="YOLO"):
        
        ok, frame = cap.read()
        if not ok:
            break

        # Для каждого кадра применяем модель
        res = model(frame, conf=conf, classes=[0], verbose=False)[0]
       
        if res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                score = float(b.conf[0]) if getattr(b, "conf", None) is not None else None

                # Рисуем зеленый bbox
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Текст: класс (Person) и conf под ним
                cls_name = "Person"
                text1 = cls_name
                text2 = f"{score:.2f}" if score is not None else ""

                # Координаты для текста
                offset_y = 18
                cv2.putText(
                    frame, text1, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame, text2, (x1, max(20, y1 + offset_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
                )
        out.write(frame)
