import cv2
import numpy as np
import os
import time
from pylibdmtx.pylibdmtx import decode
from concurrent.futures import ThreadPoolExecutor


def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def detect_black_regions(image_path):
    start_time = time.perf_counter()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Ошибка загрузки изображения. Проверь путь!")

    image = sharpen_image(image)
    (B, G, R) = cv2.split(image)
    threshold = 70
    black_mask = np.uint8((B < threshold) & (G < threshold) & (R < threshold)) * 255
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]
    result_image = image.copy()

    for dir_name in ['tmp', 'detected', 'processed']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        else:
            clear_directory(dir_name)

    crop_data = []

    detected_image = image.copy()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if B[y, x] < threshold and G[y, x] < threshold and R[y, x] < threshold:
                detected_image[y, x] = [0, 0, 255]

    cv2.imwrite("detected/detected_image.jpg", detected_image)

    for idx, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        x, y = max(0, x - 100), max(0, y - 100)
        w, h = min(image.shape[1] - x, w + 200), min(image.shape[0] - y, h + 200)
        cropped_image = image[y:y + h, x:x + w]
        crop_path = f"tmp/cropped_image_{idx + 1}.jpg"
        cv2.imwrite(crop_path, cropped_image)
        crop_data.append((crop_path, x, y, w, h))

        cx, cy = int(x + w / 2), int(y + h / 2)
        cv2.circle(detected_image, (cx, cy), 10, (0, 0, 255), -1)

    cv2.imwrite("detected/detected_image.jpg", detected_image)

    recognized_regions, unique_codes = process_images_from_tmp(crop_data)

    for x, y, w, h in recognized_regions:
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("processed/processed_image.jpg", result_image)
    timer(start_time)


def timer(start_time):
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) * 1000
    minutes = int(elapsed_time // 60000)
    seconds = int((elapsed_time % 60000) // 1000)
    milliseconds = int(elapsed_time % 1000)
    print(f"Обработка завершена за {minutes} мин {seconds} сек {milliseconds} мс")

def decode_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return []

    decoded_objects = decode(image)

    return [obj.data.decode('utf-8') for obj in decoded_objects] if decoded_objects else []


def process_images_from_tmp(crop_data):
    results = set()
    recognized_regions = []

    with ThreadPoolExecutor() as executor:
        decoded_lists = executor.map(decode_image, [crop[0] for crop in crop_data])

    for (crop_path, x, y, w, h), decoded_list in zip(crop_data, decoded_lists):
        if decoded_list:
            results.update(decoded_list)
            recognized_regions.append((x, y, w, h))

    if results:
        with open("processed/results2.txt", "w") as file:
            file.write("\n".join(results))
    else:
        print("DataMatrix метки не найдены!")

    return recognized_regions, results


detect_black_regions("test1.jpg")
