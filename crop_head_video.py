import math
import os
import sys
import cv2
import json
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'data_utils'))
from detect_face import SCRFD


class HeadTracker:
    def __init__(self, pan_alpha=0.15, zoom_alpha=0.08, dead_zone_ratio=0.35,
                 pad_ratio_range=(1.8, 2.6), min_face_size=30, output_size=848):
        self.pan_alpha = pan_alpha
        self.zoom_alpha = zoom_alpha
        self.dead_zone_ratio = dead_zone_ratio
        self.pad_ratio_range = pad_ratio_range
        self.min_face_size = min_face_size
        self.output_size = output_size
        self.default_pad_ratio = sum(pad_ratio_range) / 2.0

        self.crop_cx = None
        self.crop_cy = None
        self.smooth_crop_size = None
        self.first_detection = False
        self.last_valid = None

    def update(self, face_cx=None, face_cy=None, face_size=0, face_w=0, face_h=0,
               frame_w=1920, frame_h=1080, face_detected=True):
        fh, fw = frame_h, frame_w

        # 面部检测丢失或面部尺寸过小
        if not face_detected or face_size < self.min_face_size:
            if self.last_valid is not None:
                return self.last_valid
            # 无历史数据，返回默认裁切框
            default_size = min(fh, fw)
            return {
                'crop_x': (fw - default_size) / 2.0,
                'crop_y': (fh - default_size) / 2.0,
                'crop_size': float(default_size),
                'face_cx': fw / 2.0,
                'face_cy': fh / 2.0,
                'face_w': 0.0,
                'face_h': 0.0,
                'smooth_crop_size': float(default_size),
                'pad_ratio': self.default_pad_ratio,
            }

        # 首次检测到面部
        if not self.first_detection:
            self.crop_cx = float(face_cx)
            self.crop_cy = float(face_cy)
            self.smooth_crop_size = float(face_size * self.default_pad_ratio)
            self.first_detection = True
        else:
            # 死区跟随逻辑
            offset_x = face_cx - self.crop_cx
            offset_y = face_cy - self.crop_cy
            offset_dist = math.sqrt(offset_x ** 2 + offset_y ** 2)
            dead_zone = self.smooth_crop_size * self.dead_zone_ratio

            if offset_dist > dead_zone and offset_dist > 0:
                overshoot = offset_dist - dead_zone
                target_cx = self.crop_cx + (offset_x / offset_dist) * overshoot
                target_cy = self.crop_cy + (offset_y / offset_dist) * overshoot
                self.crop_cx = self.pan_alpha * target_cx + (1 - self.pan_alpha) * self.crop_cx
                self.crop_cy = self.pan_alpha * target_cy + (1 - self.pan_alpha) * self.crop_cy

            # 动态 pad_ratio 与焦距平滑
            target_crop_size = face_size * self.default_pad_ratio
            self.smooth_crop_size = self.zoom_alpha * target_crop_size + (1 - self.zoom_alpha) * self.smooth_crop_size
            self.smooth_crop_size = max(face_size * self.pad_ratio_range[0],
                                        min(self.smooth_crop_size, face_size * self.pad_ratio_range[1]))

        # 画面边界约束
        crop_size = self.smooth_crop_size
        crop_size = min(crop_size, min(fh, fw))

        crop_cx = max(crop_size / 2.0, min(self.crop_cx, fw - crop_size / 2.0))
        crop_cy = max(crop_size / 2.0, min(self.crop_cy, fh - crop_size / 2.0))

        crop_x = crop_cx - crop_size / 2.0
        crop_y = crop_cy - crop_size / 2.0

        actual_pad_ratio = crop_size / face_size if face_size > 0 else self.default_pad_ratio

        result = {
            'crop_x': float(crop_x),
            'crop_y': float(crop_y),
            'crop_size': float(crop_size),
            'face_cx': float(face_cx),
            'face_cy': float(face_cy),
            'face_w': float(face_w),
            'face_h': float(face_h),
            'smooth_crop_size': float(self.smooth_crop_size),
            'pad_ratio': float(actual_pad_ratio),
        }
        self.last_valid = result
        return result


def detect_face(detector, frame):
    bboxes, indices, kpss = detector.detect(frame)
    if len(indices) == 0:
        return None
    best_idx = indices[0]
    scores = bboxes[:, 4] if bboxes.shape[1] > 4 else None
    if scores is not None and len(indices) > 1:
        best_idx = indices[0]
        for idx in indices:
            if bboxes[idx, 4] > bboxes[best_idx, 4]:
                best_idx = idx

    x1 = int(bboxes[best_idx, 0])
    y1 = int(bboxes[best_idx, 1])
    w = int(bboxes[best_idx, 2])
    h = int(bboxes[best_idx, 3])
    conf = float(bboxes[best_idx, 4]) if bboxes.shape[1] > 4 else 1.0
    return x1, y1, w, h, conf


def crop_and_resize(frame, crop_info, output_size=848):
    fh, fw = frame.shape[:2]
    x1 = int(round(crop_info['crop_x']))
    y1 = int(round(crop_info['crop_y']))
    cs = int(round(crop_info['crop_size']))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(fw, x1 + cs)
    y2 = min(fh, y1 + cs)

    cropped = frame[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)
    scale_factor = output_size / cs
    return resized, scale_factor


def process_video(video_path, output_dir, onnx_model_path, output_size=848,
                  pan_alpha=0.15, zoom_alpha=0.08, dead_zone_ratio=0.35,
                  pad_ratio_range=(1.8, 2.6), min_face_size=30,
                  conf_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    silent_video_path = os.path.join(output_dir, f'{base_name}_head_{output_size}_silent.mp4')
    output_video_path = os.path.join(output_dir, f'{base_name}_head_{output_size}.mp4')
    metadata_path = os.path.join(output_dir, f'{base_name}_head_{output_size}_metadata.json')

    detector = SCRFD(onnx_model_path, confThreshold=conf_threshold, nmsThreshold=0.5)
    tracker = HeadTracker(
        pan_alpha=pan_alpha, zoom_alpha=zoom_alpha,
        dead_zone_ratio=dead_zone_ratio, pad_ratio_range=pad_ratio_range,
        min_face_size=min_face_size, output_size=output_size,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video: {video_path}')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'[INFO] Video: {video_path}')
    print(f'[INFO] Resolution: {frame_w}x{frame_h}, FPS: {fps:.2f}, Frames: {total_frames}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(silent_video_path, fourcc, fps, (output_size, output_size))

    metadata = {
        'source_video': video_path,
        'output_video': output_video_path,
        'output_size': output_size,
        'fps': fps,
        'source_resolution': [frame_w, frame_h],
        'pan_alpha': pan_alpha,
        'zoom_alpha': zoom_alpha,
        'dead_zone_ratio': dead_zone_ratio,
        'pad_ratio_range': list(pad_ratio_range),
        'frames': []
    }

    print('[INFO] Processing frames...')
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        face_result = detect_face(detector, frame)

        if face_result is not None:
            x1, y1, w, h, conf = face_result
            face_size = max(w, h)
            face_cx = x1 + w / 2.0
            face_cy = y1 + h / 2.0
            crop_info = tracker.update(
                face_cx=face_cx, face_cy=face_cy, face_size=face_size,
                face_w=w, face_h=h, frame_w=frame_w, frame_h=frame_h,
            )
        else:
            crop_info = tracker.update(
                face_detected=False, frame_w=frame_w, frame_h=frame_h,
            )

        resized, scale_factor = crop_and_resize(frame, crop_info, output_size)
        writer.write(resized)

        frame_meta = {
            'frame_idx': frame_idx,
            'crop_x': crop_info['crop_x'],
            'crop_y': crop_info['crop_y'],
            'crop_size': crop_info['crop_size'],
            'scale_factor': scale_factor,
            'face_detected': face_result is not None,
            'pad_ratio': crop_info.get('pad_ratio', 0),
            'face_bbox': {
                'x': crop_info['face_cx'] - crop_info['face_w'] / 2,
                'y': crop_info['face_cy'] - crop_info['face_h'] / 2,
                'w': crop_info['face_w'],
                'h': crop_info['face_h'],
            } if face_result else None,
            'smooth_center': {
                'cx': crop_info.get('face_cx', frame_w / 2),
                'cy': crop_info.get('face_cy', frame_h / 2),
            },
        }
        metadata['frames'].append(frame_meta)

    cap.release()
    writer.release()

    print('[INFO] Merging audio from original video...')
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', silent_video_path,
        '-i', video_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'aac', '-b:a', '192k',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-shortest',
        output_video_path
    ]
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print('[INFO] Audio merged successfully.')
            if os.path.exists(output_video_path):
                os.remove(silent_video_path)
                print('[INFO] Temporary silent video removed.')
        else:
            print(f'[WARN] ffmpeg merge failed, keeping silent video.')
            print(f'[WARN] ffmpeg stderr: {result.stderr[:500]}')
            if os.path.exists(silent_video_path):
                import shutil
                shutil.move(silent_video_path, output_video_path)
    except subprocess.TimeoutExpired:
        print('[WARN] ffmpeg timed out, keeping silent video.')
        if os.path.exists(silent_video_path):
            import shutil
            shutil.move(silent_video_path, output_video_path)
    except FileNotFoundError:
        print('[WARN] ffmpeg not found, keeping silent video.')
        if os.path.exists(silent_video_path):
            import shutil
            shutil.move(silent_video_path, output_video_path)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f'[INFO] Output video: {output_video_path}')
    print(f'[INFO] Metadata: {metadata_path}')
    print(f'[INFO] Total frames processed: {len(metadata["frames"])}')

    return output_video_path, metadata_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and crop head region from video (PTZ mode)')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as video directory)')
    parser.add_argument('--onnx_model', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data_utils', 'scrfd_2.5g_kps.onnx'),
                        help='Path to SCRFD ONNX model')
    parser.add_argument('--output_size', type=int, default=848,
                        help='Output video size (default: 848)')
    parser.add_argument('--pan_alpha', type=float, default=0.15,
                        help='EMA alpha for camera pan (higher = more responsive, default: 0.15)')
    parser.add_argument('--zoom_alpha', type=float, default=0.08,
                        help='EMA alpha for camera zoom (higher = more responsive, default: 0.08)')
    parser.add_argument('--dead_zone_ratio', type=float, default=0.35,
                        help='Dead zone ratio as fraction of crop size (default: 0.35)')
    parser.add_argument('--pad_ratio_min', type=float, default=1.8,
                        help='Minimum pad ratio (face close, default: 1.8)')
    parser.add_argument('--pad_ratio_max', type=float, default=2.6,
                        help='Maximum pad ratio (face far, default: 2.6)')
    parser.add_argument('--min_face_size', type=int, default=30,
                        help='Minimum face size to consider valid (default: 30)')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Face detection confidence threshold (default: 0.5)')

    opt = parser.parse_args()

    if opt.output_dir is None:
        opt.output_dir = os.path.dirname(opt.video_path)

    process_video(
        video_path=opt.video_path,
        output_dir=opt.output_dir,
        onnx_model_path=opt.onnx_model,
        output_size=opt.output_size,
        pan_alpha=opt.pan_alpha,
        zoom_alpha=opt.zoom_alpha,
        dead_zone_ratio=opt.dead_zone_ratio,
        pad_ratio_range=(opt.pad_ratio_min, opt.pad_ratio_max),
        min_face_size=opt.min_face_size,
        conf_threshold=opt.conf_threshold,
    )
