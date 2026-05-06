# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Video static segment detection')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing videos to process')
    return parser.parse_args()


def has_static_segment(
    video_path,
    min_static_duration=5.0,
    diff_threshold=0.1,
) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"fail to open the video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_static_frames = int(min_static_duration * fps)

    prev_gray = None
    consecutive_static_frames = 0

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = np.mean(diff)

            if mean_diff < diff_threshold:
                consecutive_static_frames += 1
                if consecutive_static_frames >= min_static_frames:
                    cap.release()
                    with open("logs/static_videos.log", "a") as f:
                        f.write(video_path + "\n")
                    return True
            else:
                consecutive_static_frames = 0

        prev_gray = gray

    cap.release()
    return False

def main():
    args = parse_args()
    dir = args.dir
    video_folders = os.listdir(dir)
    videos_to_be_verified = []
    for video_folder in video_folders:
        video_path = os.path.join(dir, video_folder)
        if os.path.isdir(video_path):
            video_files = os.listdir(video_path)
            for video_file in video_files:
                video_file_path = os.path.join(video_path, video_file)
                videos_to_be_verified.append(video_file_path)

    max_workers = 64
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(has_static_segment, videos_to_be_verified), total=len(videos_to_be_verified), desc="Verifying videos"))

if __name__ == "__main__":
    main()