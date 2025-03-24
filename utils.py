import os
import time
import cv2


def clear_or_create():
    for dir_name in ['tmp', 'detected', 'processed']:
        os.makedirs(dir_name, exist_ok=True)

        for filename in os.listdir(dir_name):
            file_path = os.path.join(dir_name, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)


def timer(start_time):
    elapsed_time = (time.perf_counter() - start_time) * 1000
    minutes, seconds = divmod(int(elapsed_time // 1000), 60)
    milliseconds = int(elapsed_time % 1000)
    print(f"Обработка завершена за {minutes} мин {seconds} сек {milliseconds} мс")


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)