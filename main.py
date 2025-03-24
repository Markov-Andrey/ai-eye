import time
import os
from concurrent.futures import ThreadPoolExecutor
from utils import clear_or_create, timer
from image_processing import process_image
from decoder import decode_matrix, decode_qr


def process_images_from_detected():
    detected_dir = "detected"
    all_unique_values = set()

    image_paths = [os.path.join(detected_dir, f) for f in os.listdir(detected_dir) if
                   f.endswith(('.png', '.jpg', '.jpeg'))]

    image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    with ThreadPoolExecutor() as executor:
        decoded_results = list(executor.map(decode_matrix, image_paths))

    all_unique_values.update([data[0] for data in decoded_results if data])

    if all_unique_values:
        os.makedirs("processed", exist_ok=True)
        with open("processed/results2.txt", "w") as file:
            file.write('\n'.join(sorted(all_unique_values)))

    return all_unique_values


if __name__ == "__main__":
    clear_or_create()
    start_time = time.perf_counter()
    image_directory = 'image'
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        process_image(image_path)

    process_images_from_detected()
    decode_qr()
    timer(start_time)
