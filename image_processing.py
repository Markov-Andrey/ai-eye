import cv2
import numpy as np
import time
import os
from utils import save_image


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def filter_rectangles(contours):
    valid_rects = []
    areas = []
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if 4 <= len(approx) <= 10:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            if w > 50 and h > 50 and 0.7 < w / h < 1.6:
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


def extract_detected_objects(image, valid_rects, areas):
    mean_area = np.mean(areas)
    std_dev = np.std(areas)

    for idx, (rect, cnt) in enumerate(valid_rects):
        (x, y), (w, h), angle = rect
        area = w * h

        if (mean_area - 1.5 * std_dev) < area < (mean_area + 1.5 * std_dev):
            M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            x1, y1 = max(0, int(x - w / 2)), max(0, int(y - h / 2))
            x2, y2 = min(image.shape[1], int(x + w / 2)), min(image.shape[0], int(y + h / 2))

            cropped = rotated[y1:y2, x1:x2]

            if cropped is None or cropped.size == 0:
                continue

            timestamp = int(time.time() * 1000)
            cropped_image_path = os.path.join('detected', f'crop_{timestamp}.png')

            save_image(cropped, cropped_image_path)


def process_image(image_path):
    print(f"Обрабатываю: {image_path}")
    timestamp = time.time_ns()
    image = cv2.imread(image_path)
    edges = preprocess_image(image)
    edges_path = f'tmp/matrix_edges_{timestamp}.jpg'
    save_image(edges, edges_path)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_rects, areas = filter_rectangles(contours)
    output_path = f'tmp/matrix_{timestamp}.jpg'
    output_image = draw_contours(image, valid_rects)
    save_image(output_image, output_path)

    if areas:
        extract_detected_objects(image, valid_rects, areas)
