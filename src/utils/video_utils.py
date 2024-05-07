import os
import pathlib

import cv2
import numpy as np


def extract_mp4(mp4_path, path_frame_dir):
    cap = cv2.VideoCapture(mp4_path)
    frame_index = 0

    if not os.path.isdir(path_frame_dir):
        pathlib.Path(path_frame_dir).mkdir(parents=True)

    print("start extracting mp4...")
    while True:
        success, frame = cap.read()
        if not success:
            break
        cv2.imwrite(f"{path_frame_dir}/{str(frame_index).zfill(10)}.png", frame)
        frame_index += 1
    print("finished extracting mp4")

    cap.release()
