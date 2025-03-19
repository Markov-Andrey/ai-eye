import cv2
import numpy as np
import time
from pylibdmtx.pylibdmtx import decode
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageEnhance, ImageFilter


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Файл {image_path} не найден или не может быть открыт.")
    return image


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.9)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_rectangles(contours):
    valid_rects = []
    areas = []
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            if w > 20 and h > 20 and 0.8 < w / h < 1.3:
                area = w * h
                areas.append(area)
                valid_rects.append((rect, cnt))
    return valid_rects, areas


def draw_contours(image, valid_rects):
    output_image = image.copy()
    for rect, _ in valid_rects:
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        cv2.drawContours(output_image, [box], 0, (0, 0, 255), 2)
    return output_image


def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image


def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def binary_threshold(image, threshold_value=128):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    return binary_image


def extract_detected_objects(image, valid_rects, areas):
    mean_area = np.mean(areas)
    std_dev = np.std(areas)

    for idx, (rect, cnt) in enumerate(valid_rects):
        (x, y), (w, h), angle = rect
        area = w * h

        if (mean_area - 1.5 * std_dev) < area < (mean_area + 1.5 * std_dev):
            M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cropped = rotated[y1:y2, x1:x2]
            binary_image = binary_threshold(cropped)
            sharpened_image = sharpen_image(binary_image)
            pil_image = Image.fromarray(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.5)
            pil_image = pil_image.filter(ImageFilter.SHARPEN)
            processed_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            cropped_image_path = os.path.join('detected', f'crop_{idx}.png')
            cv2.imwrite(cropped_image_path, processed_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def process_image(image_path):
    image = load_image(image_path)
    edges = preprocess_image(image)
    save_image(edges, 'tmp/edges_image.jpg')
    contours = find_contours(edges)
    valid_rects, areas = filter_rectangles(contours)
    output_image = draw_contours(image, valid_rects)
    save_image(output_image, 'tmp/output_image.jpg')
    if areas:
        extract_detected_objects(image, valid_rects, areas)
    print("Границы: tmp/edges_image.jpg")
    print("Контуры: tmp/output_image.jpg")


def timer(start_time):
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) * 1000
    minutes = int(elapsed_time // 60000)
    seconds = int((elapsed_time % 60000) // 1000)
    milliseconds = int(elapsed_time % 1000)
    print(f"Обработка завершена за {minutes} мин {seconds} сек {milliseconds} мс")


def process_images_from_detected():
    detected_dir = "detected"
    results = {}

    image_paths = [os.path.join(detected_dir, f) for f in os.listdir(detected_dir) if
                   f.endswith(('.png', '.jpg', '.jpeg'))]

    # сортировка
    image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    with ThreadPoolExecutor() as executor:
        decoded_lists = executor.map(decode_image, image_paths)

    for image_path, decoded_list in zip(image_paths, decoded_lists):
        if decoded_list:
            results[os.path.basename(image_path)] = decoded_list

    if results:
        os.makedirs("processed", exist_ok=True)
        with open("processed/results2.txt", "w") as file:
            for name, value in results.items():
                file.write(f"{name}: {' '.join(value)}\n")

    return results


def decode_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return []

    decoded_objects = decode(image)

    return [obj.data.decode('utf-8') for obj in decoded_objects] if decoded_objects else []


import os


def clear_or_create():
    for dir_name in ['tmp', 'detected', 'processed']:
        os.makedirs(dir_name, exist_ok=True)

        for filename in os.listdir(dir_name):
            file_path = os.path.join(dir_name, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)


if __name__ == "__main__":
    clear_or_create()
    start_time = time.perf_counter()
    process_image('test1.jpg')
    process_images_from_detected()
    timer(start_time)
