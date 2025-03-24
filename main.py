import time
import os
from utils import clear_or_create, timer
from image_processing import process_image
from decoder import decode_matrix, decode_qr


if __name__ == "__main__":
    clear_or_create()
    start_time = time.perf_counter()
    image_directory = 'image'
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        process_image(image_path)

    decode_matrix()
    decode_qr()
    timer(start_time)
