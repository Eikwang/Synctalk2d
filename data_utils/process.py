import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm


def extract_audio(path, out_path, sample_rate=16000):
    """Extract audio from video file to WAV.

    Args:
        path: Path to video file
        out_path: Output WAV path
        sample_rate: Target sample rate (default 16000)
    """
    # AC-021: aud.wav 已存在且非空时跳过
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f'[INFO] Audio already exists at {out_path}, skipping extraction.')
        return

    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')


def extract_images(path, base_dir=None):
    """Extract frames from video file.

    Args:
        path: Path to video file
        base_dir: Output directory (defaults to video file's parent directory)
    """
    if base_dir is None:
        base_dir = os.path.dirname(path)

    full_body_dir = os.path.join(base_dir, "full_body_img")
    # AC-021: full_body_img/ 已存在且帧数 > 0 时跳过
    if os.path.exists(full_body_dir) and len(os.listdir(full_body_dir)) > 0:
        print(f'[INFO] Images already exist in {full_body_dir}, skipping extraction.')
        return
    if not os.path.exists(full_body_dir):
        os.mkdir(full_body_dir)

    counter = 0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        cmd = f'ffmpeg -i {path} -vf "fps=25" -c:v libx264 -c:a aac {path.replace(".mp4", "_25fps.mp4")}'
        os.system(cmd)
        path = path.replace(".mp4", "_25fps.mp4")

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        raise ValueError("Your video fps should be 25!!!")

    print("extracting images...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(full_body_dir, str(counter)+'.jpg'), frame)
        counter += 1


def get_audio_feature(wav_path):
    """Extract audio features using AudioEncoder model.

    Args:
        wav_path: Path to WAV file (output will be {wav_path}_ave.npy)
    """
    print("extracting audio feature...")
    os.system("python ./data_utils/ave/test_w2l_audio.py --wav_path "+wav_path)


def get_landmark(base_dir, landmarks_dir):
    """Detect facial landmarks for all frames in full_body_img.

    Args:
        base_dir: Directory containing full_body_img/
        landmarks_dir: Output directory for landmark files
    """
    print("detecting landmarks...")
    full_img_dir = os.path.join(base_dir, "full_body_img")

    from get_landmark import Landmark
    landmark = Landmark()

    # Numeric sort to ensure correct frame order
    img_files = sorted(
        [f for f in os.listdir(full_img_dir) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    for img_name in tqdm(img_files):
        img_path = os.path.join(full_img_dir, img_name)
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        pre_landmark, x1, y1 = landmark.detect(img_path)
        with open(lms_path, "w") as f:
            for p in pre_landmark:
                x, y = p[0]+x1, p[1]+y1
                f.write(str(x))
                f.write(" ")
                f.write(str(y))
                f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file or preprocessed directory")
    opt = parser.parse_args()

    # Determine if path is a directory (crop_head_video output) or video file
    if os.path.isdir(opt.path):
        # Directory mode: use as base_dir directly
        base_dir = opt.path
        video_path = None
        # Find video file in directory
        for f in os.listdir(base_dir):
            if f.endswith('.mp4'):
                video_path = os.path.join(base_dir, f)
                break
        if video_path is None:
            print(f"[INFO] No video file found in {base_dir}, skipping video-based extraction.")
    else:
        # Video file mode: derive base_dir from video path
        video_path = opt.path
        base_dir = os.path.dirname(opt.path)

    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)

    # Skip extract_audio/extract_images if directory mode (already done by crop_head_video)
    if os.path.isdir(opt.path):
        print(f"[INFO] Directory mode: skipping video extraction (assuming crop_head_video.py already processed)")
    else:
        extract_audio(video_path, wav_path)
        extract_images(video_path, base_dir)

    get_landmark(base_dir, landmarks_dir)
    get_audio_feature(wav_path)
    
    