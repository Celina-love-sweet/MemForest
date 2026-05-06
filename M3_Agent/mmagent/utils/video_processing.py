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
import base64
import logging
import os
import tempfile
import math
import cv2
import numpy as np
from moviepy import VideoFileClip
import subprocess

# Disable moviepy logging
logging.getLogger('moviepy').setLevel(logging.ERROR)
# Disable moviepy's tqdm progress bar
logging.getLogger('moviepy.video.io.VideoFileClip').setLevel(logging.ERROR)
logging.getLogger('moviepy.audio.io.AudioFileClip').setLevel(logging.ERROR)

# Configure logging
logger = logging.getLogger(__name__)

def get_video_info(file_path):
    """Get video/audio information using appropriate libraries.
    
    Args:
        file_path (str): Path to video or audio file
        
    Returns:
        dict: Dictionary containing media metadata
    """
    file_info = {}
    file_info["path"] = file_path
    file_info["name"] = file_path.split("/")[-1]
    file_info["format"] = os.path.splitext(file_path)[1][1:].lower()
        
    # Handle video files using moviepy
    
    video = VideoFileClip(file_path)  # Disable logging for this instance
    
    # Get basic properties from moviepy
    file_info["fps"] = video.fps
    file_info["frames"] = int(video.fps * video.duration)
    file_info["duration"] = video.duration
    file_info["width"] = video.size[0]
    file_info["height"] = video.size[1]
    
    video.close()
    return file_info

def extract_frames(video, start_time=None, interval=None, sample_fps=10):
    # if start_time and interval are not provided, sample the whole video at sample_fps
    if start_time is None and interval is None:
        start_time = 0
        interval = video.duration

    frames = []
    frame_interval = 1.0 / sample_fps

    # Extract frames at specified intervals
    for t in np.arange(
        start_time, min(start_time + interval, video.duration), frame_interval
    ):
        frame = video.get_frame(t)
        # Convert frame to jpg and base64
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(base64.b64encode(buffer).decode("utf-8"))
        
    return frames

# TODO: check if there is a better way to do this without repeatedly opening and closing the video file
def process_video_clip(video_path, fps=5, audio_fps=16000): 
    try: 
        base64_data = {}
        video = VideoFileClip(video_path)
        base64_data["video"] = base64.b64encode(open(video_path, "rb").read())
        base64_data["frames"] = extract_frames(video, sample_fps=fps)
        
        if video.audio is None:
            base64_data["audio"] = None
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav") as audio_tempfile:
                video.audio.write_audiofile(audio_tempfile.name, codec="pcm_s16le", fps=audio_fps)
                audio_tempfile.seek(0)
                base64_data["audio"] = base64.b64encode(audio_tempfile.read())
        
        video.close()
        return base64_data["video"], base64_data["frames"], base64_data["audio"]

    except Exception as e:
        logger.error(f"Error processing video clip: {str(e)}")
        raise

def verify_video_processing(video_path, output_dir, interval, strict=False):
    """Verify that a video was properly split into clips by checking the number of clips.
    
    Args:
        video_path (str): Path to original video file
        output_dir (str): Directory containing the split clips
        interval (float): Interval length in seconds used for splitting
        
    Returns:
        bool: True if verification passes, False otherwise
    """

    def has_video_and_audio(file_path):
        def has_stream(stream_type):
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", stream_type,
                "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                capture_output=True, text=True
            )
            return bool(result.stdout.strip())

        return has_stream("v:0") and has_stream("a:0")

    def has_static_segment(
        video_path,
        min_static_duration=5.0,
        diff_threshold=0.001,
    ) -> bool:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_path}")

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
                        return True
                else:
                    consecutive_static_frames = 0

            prev_gray = gray

        cap.release()
        return False

    try:
        if not os.path.exists(video_path):
            with open("logs/video_processing_failed.log", "a") as f:
                f.write(f"Error processing {video_path}: Video file not found.\n")
            logger.error(f"Error processing {video_path}: Video file not found.")
            return False
        # Get expected number of clips based on video duration
        video_info = get_video_info(video_path)
        expected_clips_num = math.ceil(int(video_info["duration"]) / interval)
        
        # Get actual number of clips in output directory
        clip_dir = output_dir
        
        if not os.path.exists(clip_dir):
            with open("logs/video_processing_failed.log", "a") as f:
                f.write(f"Error processing {video_path}: Clip directory {clip_dir} not found.\n")
            logger.error(f"Error processing {video_path}: Clip directory {clip_dir} not found.")
            return False
            
        actual_clips = [f for f in os.listdir(clip_dir) if os.path.isfile(os.path.join(clip_dir, f)) and f.split('.')[-1] in ['mp4', 'mov', 'webm']]
        actual_clips_num = len(actual_clips)
        
        if actual_clips_num != expected_clips_num:
            with open("logs/video_processing_failed.log", "a") as f:
                f.write(f"Error processing {video_path}: Expected {video_info['duration']}/{interval}={expected_clips_num} clips, but found {actual_clips_num} clips.\n")
            logger.error(f"Error processing {video_path}: Expected {video_info['duration']}/{interval}={expected_clips_num} clips, but found {actual_clips_num} clips.")
            return False

        if strict:
            clip_files = [os.path.join(clip_dir, clip) for clip in actual_clips]
            for clip_file in clip_files:
                clip_id = clip_file.split("/")[-1].split(".")[0]
                if not has_video_and_audio(clip_file):
                    with open("logs/video_processing_failed.log", "a") as f:
                        f.write(f"Error processing {clip_file}: No video or audio streams found.\n")
                    logger.error(f"Error processing {clip_file}: No video or audio streams found.")
                    return False
                if int(clip_id) < len(clip_files)-2 and has_static_segment(clip_file):
                    with open("logs/video_processing_failed.log", "a") as f:
                        f.write(f"Error processing {clip_file}: Has static segment.\n")
                    logger.error(f"Error processing {clip_file}: Has static segment.")
                    return False

        return True
        
    except Exception as e:
        with open("logs/video_processing_failed.log", "a") as f:
            f.write(f"Error verifying {video_path}: {e}\n")
        logger.error(f"Error verifying {video_path}: {e}")
        return False
