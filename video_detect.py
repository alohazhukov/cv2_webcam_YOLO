# Обработка видео с помощью YOLO

import cv2
import numpy as np
from utils import FPS_counter, show_fps
from ultralytics import YOLO
from patched_yolo_infer import (
    visualize_results_usual_yolo_inference,
)


# Инициализируем модель для детекции
# Для детекции можно использовать модель yolov8m.pt с параметрами (segment=False, show_boxes=True)
model = YOLO("yolov8m.pt")

# Зададим параметры для дальнейшего использования размер/уверенность/отсеевание дубликатов
imgsz = 640
conf = 0.4
iou = 0.7

# Инициализация, указываем путь к видео
cap = cv2.VideoCapture('видео')

# Усредняем FPS
fps_counter = FPS_counter(calc_time_perion_N_frames=10)

# Проверяем открылось ли видео
if not cap.isOpened():
    print("Не удалось открыть видео")
    exit()

# Получение исходного FPS и размеров кадра
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создание объекта VideoWriter для сохранения обработанного видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))


while True:
    # Захват кадра
    ret, frame = cap.read()

    # Проверка, успешно ли захвачен кадр
    if not ret:
        print("Не удалось получить кадр из видео")
        break

    frame = visualize_results_usual_yolo_inference(
        frame,
        model,
        imgsz,
        conf,
        iou,
        segment=False,
        delta_colors=3,
        thickness=4,
        font_scale=1.5,
        show_boxes=True,
        fill_mask=True,
        alpha=0.2,
        show_confidences=True,
        return_image_array=True
    )

    # Ресайз
    scale = 0.5
    frame_resized = cv2.resize(frame, (-1, 1), fx=scale, fy=scale)

    frame_resized = show_fps(frame_resized, fps_counter)

    # Отображение кадра
    cv2.imshow('видео', frame_resized)

    # Запись/сохранение в выходной видеофайл (без FPS и ресайза)
    out.write(frame)

    # Остановка/выход из цикла по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие окна
cap.release()
out.release()
cv2.destroyAllWindows()
