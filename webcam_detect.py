# pip install patched-yolo-infer

import cv2
import numpy as np
from ultralytics import YOLO
from patched_yolo_infer import (
    visualize_results_usual_yolo_inference,
)
from utils import FPS_counter, show_fps


# Инициализируем сегментационную модель YOLOv8m-seg
# Для детекции можно использовать модель yolov8m.pt с параметрами (segment=False, show_boxes=True)
# Либо использовать нано yolov8n - эта модель будет полегче
model = YOLO("yolov8m-seg.pt")

# Зададим параметры для дальнейшего использования размер/уверенность/отсеевание дубликатов
imgsz = 640
conf = 0.35
iou = 0.7

# Усредняем FPS
fps_counter = FPS_counter(calc_time_perion_N_frames=10)

# Инициализация веб-камеры cv2.VideoCapture('номер камеры') в зависимости от кол-ва подклченных камер 0,1,2 и тд
cap = cv2.VideoCapture(0)

# Проверяем запущенна ли камера
if not cap.isOpened():
    print("Не удалось открыть камеру")

# Установка разрешения на 720p (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Захват кадра с веб-камеры
    ret, frame = cap.read()

    # Проверка, успешно ли захвачен кадр
    if not ret:
        print("Не удалось получить кадр с веб-камеры")
        break

    frame = visualize_results_usual_yolo_inference(
        frame,
        model,
        imgsz,
        conf,
        iou,
        segment=True,
        delta_colors=3,
        thickness=8,
        font_scale=1.5,
        show_boxes=False,
        fill_mask=True,
        alpha=0.2,
        show_confidences=True,
        return_image_array=True
    )

    # Ресайз
    scale = 0.5
    frame = cv2.resize(frame, (-1, 1), fx=scale, fy=scale)

    frame = show_fps(frame, fps_counter)

    # Отображение кадра
    cv2.imshow('Webcam', frame)

    # Остановка/выход из цикла по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие окна
cap.release()
cv2.destroyAllWindows()
