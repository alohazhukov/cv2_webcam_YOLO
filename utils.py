# Функции для вычисления и отображения FPS

import time
import numpy as np
import cv2


class FPS_counter:

    def __init__(self, calc_time_perion_N_frames: int) -> None:
        """
        Счетчик FPS по ограниченным участкам видео (скользящему окну)

        Args: calc_time_perion_N_frames (int) - количество фреймов окна для подсчета статистики.
        """
        self.time_buffer = []
        self.calc_time_perion_N_frames = calc_time_perion_N_frames

    def calc_FPS(self) -> float:
        """Вычесляет FPS по нескольким кадрам

        Return: float (количество FPS)
        """
        time_buffer_is_full = len(
            self.time_buffer) == self.calc_time_perion_N_frames
        t = time.time()
        self.time_buffer.append(t)

        if time_buffer_is_full:
            self.time_buffer.pop(0)
            fps = len(self.time_buffer) / \
                (self.time_buffer[-1] - self.time_buffer[0])
            return np.round(fps, 2)
        else:
            return 0.0


def show_fps(frame, fps_counter):
    """Вычесляет и отображает FPS
    Return"""

    fps_real = fps_counter.calc_FPS()
    text = f"FPS: {fps_real:.1f}"

    # Параметры для шрифта:
    fontFace = 1
    fontScale = 1.3
    thickness = 1

    # Узнаем размер текста
    (label_width, label_height), _ = cv2.getTextSize(
        text,
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness
    )

    # Рисуем черный закрашенный прямоугольник и выводим белым цветом по верх него FPS
    frame = cv2.rectangle(frame, (0, 0), (10 + label_width,
                          15 + label_height), (0, 0, 0), -1)
    frame = cv2.putText(
        img=frame,
        text=text,
        org=(5, 20),
        fontFace=fontFace,
        fontScale=fontScale,
        thickness=thickness,
        color=(255, 255, 255)
    )
    return frame
