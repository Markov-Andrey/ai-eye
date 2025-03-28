import os
import time
from concurrent.futures import ThreadPoolExecutor
from pylibdmtx import pylibdmtx
from pyzbar import pyzbar
from PIL import Image, ImageDraw


def decode_qr():
    images = [f for f in os.listdir('image') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    unique_qrcodes = set()

    for image_file in images:
        image_path = os.path.join('image', image_file)
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        decoded_objects = pyzbar.decode(img)
        draw = ImageDraw.Draw(img)

        for obj in decoded_objects:
            rect_points = obj.polygon
            if rect_points:
                draw.polygon(rect_points, outline=(0, 255, 0), width=5)
                draw.line(rect_points + [rect_points[0]], fill=(0, 255, 0), width=5)

            qr_data = obj.data.decode()
            unique_qrcodes.add(qr_data)

        timestamp = int(time.time())
        output_image_path = os.path.join('tmp', f'qr_{timestamp}_{image_file}')
        img.save(output_image_path)

    with open('processed/results1.txt', 'w') as f:
        for qr_code in unique_qrcodes:
            f.write(qr_code + '\n')


def find_matrix(img_path):
    import cv2
    image = cv2.imread(img_path)
    decoded_objects = pylibdmtx.decode(image)

    return [obj.data.decode('utf-8') for obj in decoded_objects] if decoded_objects else []


def decode_matrix():
    detected_dir = "detected"
    all_unique_values = set()
    image_paths = [os.path.join(detected_dir, f) for f in os.listdir(detected_dir) if
                   f.endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
    with ThreadPoolExecutor() as executor:
        decoded_results = list(executor.map(find_matrix, image_paths))
    all_unique_values.update([data[0] for data in decoded_results if data])
    os.makedirs("processed", exist_ok=True)
    if all_unique_values:
        with open("processed/results2.txt", "w") as file:
            file.write('\n'.join(sorted(all_unique_values)))
    with open("processed/results3.txt", "w") as file:
        for image_path, result in zip(image_paths, decoded_results):
            if result:
                file.write(f"{os.path.basename(image_path)} - {result[0]}\n")
            else:
                file.write(f"{os.path.basename(image_path)} - false\n")

    return all_unique_values
