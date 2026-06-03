import math
import os
import sys
import subprocess
import cv2
import json
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'data_utils'))
from detect_face import SCRFD


import os
import subprocess
import cv2
import math

def preprocess_video(video_path, output_dir, base_name, target_fps=25, audio_sample_rate=16000, target_height=1080):
    """
    预处理视频：缩放、转码、统一帧率
    优化点：引入去块滤镜，降低 CRF 阈值，使用 Slow 预设以减少宏块效应。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[WARN] Cannot open video for preprocessing: {video_path}')
        return video_path, 25.0, False

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 判断是否需要缩放
    need_scale = False
    scale_filter = ''
    new_w, new_h = original_w, original_h

    # 优化：更严谨的横竖屏及分辨率判断逻辑
    if original_w > original_h:
        # 横屏：以高度为基准
        if original_h != target_height:
            need_scale = True
            new_h = target_height
            new_w = int(original_w * target_height / original_h)
            new_w = new_w + (new_w % 2)  # 确保宽度为偶数
            scale_filter = f'scale={new_w}:{new_h}:flags=lanczos'
    else:
        # 竖屏：以宽度为基准
        if original_w != target_height:
            need_scale = True
            new_w = target_height
            new_h = int(original_h * target_height / original_w)
            new_h = new_h + (new_h % 2)  # 确保高度为偶数
            scale_filter = f'scale={new_w}:{new_h}:flags=lanczos'

    need_fps = abs(original_fps - target_fps) >= 0.1

    if not need_scale and not need_fps:
        print(f'[INFO] Video already at {original_fps:.2f}fps {original_w}x{original_h}, skipping preprocessing.')
        return video_path, original_fps, False

    preprocessed_path = os.path.join(output_dir, f'{base_name}_remux_{target_fps}fps.mp4')

    # 构建优化的 FFmpeg 命令
    ffmpeg_cmd = ['ffmpeg', '-y', '-i', video_path]

    vf_filters = []
    if need_scale:
        vf_filters.append(scale_filter)

    if vf_filters:
        ffmpeg_cmd.extend(['-vf', ','.join(vf_filters)])

    ffmpeg_cmd.extend([
        '-c:v', 'libx264',
        '-preset', 'slow',     
        '-crf', '16',       
        '-pix_fmt', 'yuv420p',    
        '-r', str(target_fps),
        '-c:a', 'aac',
        '-ar', str(audio_sample_rate),
        '-ac', '1',
        '-movflags', '+faststart',
        preprocessed_path,
    ])

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0 and os.path.exists(preprocessed_path):
            changes = []
            if need_fps:
                changes.append(f'{original_fps:.2f}fps -> {target_fps}fps')
            if need_scale:
                changes.append(f'{original_w}x{original_h} -> {new_w}x{new_h}')
            print(f'[INFO] Preprocessed: {", ".join(changes)}, audio {audio_sample_rate}Hz')
            return preprocessed_path, original_fps, True
        else:
            print(f'[WARN] Preprocessing failed: {result.stderr[:300]}')
            return video_path, original_fps, False

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f'[WARN] Preprocessing error: {e}')
        return video_path, original_fps, False


def split_video_to_frames(video_path, output_dir, base_name, quality=95):
    """
    用 ffmpeg 将视频拆分为序列帧 jpg
    优化点：使用 -qmin/-qmax 锁定最高质量，启用 chroma subsampling 优化，防止二次压缩产生方块。
    """
    frames_dir = os.path.join(output_dir, 'original_img')
    os.makedirs(frames_dir, exist_ok=True)

    frame_pattern = os.path.join(frames_dir, '%d.jpg')

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-qmin', '1', '-qmax', '2',
        '-pix_fmt', 'yuvj420p',
        '-start_number', '0',
        frame_pattern,
    ]

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f'[WARN] ffmpeg split failed: {result.stderr[:300]}, falling back to cv2')
            return _split_video_to_frames_cv2(video_path, frames_dir, quality)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f'[WARN] ffmpeg split error: {e}, falling back to cv2')
        return _split_video_to_frames_cv2(video_path, frames_dir, quality)

    frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if frame_count == 0:
        print('[WARN] ffmpeg produced no frames, falling back to cv2')
        return _split_video_to_frames_cv2(video_path, frames_dir, quality)

    print(f'[INFO] Split video to {frame_count} frames in {frames_dir}')
    return frames_dir, frame_count


def _split_video_to_frames_cv2(video_path, frames_dir, quality=95):
    """OpenCV 兜底拆帧方案"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video for frame splitting: {video_path}')
        return frames_dir, 0

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f'{frame_count}.jpg')
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frame_count += 1
    cap.release()

    print(f'[INFO] Split video (cv2 fallback) to {frame_count} frames in {frames_dir}')
    return frames_dir, frame_count


def extract_audio_from_video(video_path, output_dir):
    """用 ffmpeg 从视频中提取音频为 wav

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录

    Returns:
        audio_path 音频文件路径，失败时返回 None
    """
    audio_path = os.path.join(output_dir, 'aud.wav')

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn', '-ar', '16000', '-ac', '1', '-f', 'wav',
        audio_path,
    ]

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f'[WARN] ffmpeg audio extract failed: {result.stderr[:300]}')
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f'[WARN] ffmpeg audio extract error: {e}')
        return None

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        print('[WARN] Audio extraction produced no output')
        return None

    print(f'[INFO] Extracted audio to {audio_path}')
    return audio_path


def bounce_frame_sequence(frames_dir, base_name, output_dir):
    """对序列帧做倒放拼接（正序+倒序），输出到 full_body_img/

    Args:
        frames_dir: 原始序列帧目录
        base_name: 文件基础名
        output_dir: 输出目录

    Returns:
        full_body_img 目录路径
    """
    full_body_dir = os.path.join(output_dir, 'full_body_img')
    os.makedirs(full_body_dir, exist_ok=True)

    # 获取排序后的帧文件列表（按文件名中的数字排序，避免字符串排序导致 10<2 的问题）
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                         key=lambda x: int(os.path.splitext(x)[0]))
    original_count = len(frame_files)

    # 正序：复制帧到 full_body_img/，重新编号
    for i, fname in enumerate(frame_files):
        src = os.path.join(frames_dir, fname)
        dst = os.path.join(full_body_dir, f'{i}.jpg')
        img = cv2.imread(src)
        cv2.imwrite(dst, img)

    # 倒序：复制帧到 full_body_img/，继续编号
    for i, fname in enumerate(reversed(frame_files)):
        src = os.path.join(frames_dir, fname)
        dst = os.path.join(full_body_dir, f'{original_count + i}.jpg')
        img = cv2.imread(src)
        cv2.imwrite(dst, img)

    total_count = original_count * 2
    print(f'[INFO] Bounce frame sequence: {original_count} → {total_count} frames in {full_body_dir}')
    return full_body_dir


class CropFaceEnhancer:
    def __init__(self, model_path, strength=0.4):
        self.strength = max(0.0, min(1.0, strength))
        self.session = None

        import onnxruntime
        onnxruntime.preload_dlls()
        import torch
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

        if not os.path.exists(model_path):
            print(f'[WARN] GFPGAN model not found: {model_path}, face enhancement disabled.')
            return

        try:
            import onnxruntime as ort
            providers = []
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f'[INFO] GFPGAN loaded: {model_path}, strength={self.strength}, providers={providers}')
        except ImportError:
            print('[WARN] onnxruntime not installed, face enhancement disabled.')
        except Exception as e:
            print(f'[WARN] Failed to load GFPGAN model: {e}, face enhancement disabled.')

    def enhance(self, img, face_bbox=None):
        """对面部区域做局部增强，背景保持不变

        Args:
            img: 848×848 BGR 图像
            face_bbox: 面部在 img 中的 bbox [x, y, w, h]，None 时返回原图
        """
        if self.session is None or self.strength == 0.0:
            return img
        if face_bbox is None:
            return img

        h, w = img.shape[:2]
        fx, fy, fw, fh = face_bbox
        fx, fy, fw, fh = int(fx), int(fy), int(fw), int(fh)

        # 扩展面部区域（1.5倍），确保增强覆盖面部周围
        expand = 0.5
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        new_fw = fw * (1 + expand)
        new_fh = fh * (1 + expand)
        ex = int(max(0, cx - new_fw / 2))
        ey = int(max(0, cy - new_fh / 2))
        ex2 = int(min(w, cx + new_fw / 2))
        ey2 = int(min(h, cy + new_fh / 2))

        # 裁出扩展后的面部区域
        face_region = img[ey:ey2, ex:ex2]
        if face_region.size == 0:
            return img
        region_h, region_w = face_region.shape[:2]

        # Resize 到 512×512 送入 GFPGAN
        face_512 = cv2.resize(face_region, (512, 512), interpolation=cv2.INTER_AREA)

        # BGR → RGB → float32 → normalize to [-1,1] → HWC→CHW → batch
        rgb = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        rgb = rgb.transpose(2, 0, 1)[np.newaxis]

        # ONNX inference
        output = self.session.run(None, {self.input_name: rgb})[0]

        # Post-process: denormalize → RGB→BGR → resize back to region size
        output = (output[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = cv2.resize(output, (region_w, region_h), interpolation=cv2.INTER_AREA)

        # 椭圆 mask + Gaussian blur + strength blend（仅在面部区域）
        mask = np.zeros((region_h, region_w), dtype=np.float32)
        center = (region_w // 2, region_h // 2)
        axes = (region_w // 2, region_h // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        mask = mask[:, :, np.newaxis]

        blended = (face_region.astype(np.float32) * (1 - self.strength * mask) +
                   output.astype(np.float32) * (self.strength * mask))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # 贴回原图
        result = img.copy()
        result[ey:ey2, ex:ex2] = blended

        return result


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

        # 面部尺寸独立平滑（解耦变焦目标与检测噪声）
        self.smooth_face_size = None
        self.face_ema_alpha = 0.12

        # 平移方向阻尼（防止左右来回移动时的抖动）
        self.prev_pan_dx = 0.0
        self.prev_pan_dy = 0.0
        self.direction_damp = 0.4

        self.first_detection = False
        self.last_valid = None

    @staticmethod
    def _smoothstep(edge0, edge1, x):
        """smoothstep 插值：在 [edge0, edge1] 区间内从 0 平滑过渡到 1"""
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-9)))
        return t * t * (3.0 - 2.0 * t)

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
            self.smooth_face_size = float(face_size)
            self.prev_pan_dx = 0.0
            self.prev_pan_dy = 0.0
            self.first_detection = True
        else:
            # ================================================================
            # 1. 面部尺寸平滑（每帧更新，为变焦提供去噪后的基准）
            # ================================================================
            fa = self.face_ema_alpha
            self.smooth_face_size = fa * face_size + (1 - fa) * self.smooth_face_size

            # ================================================================
            # 2. 平移跟踪（软死区 + 方向阻尼）
            # ================================================================
            offset_x = face_cx - self.crop_cx
            offset_y = face_cy - self.crop_cy
            offset_dist = math.sqrt(offset_x ** 2 + offset_y ** 2)
            dead_zone = self.smooth_crop_size * self.dead_zone_ratio

            if offset_dist > 0:
                # 软死区：smoothstep 在 [60%死区, 100%死区] 内从 0 渐变到 1
                soft_start = dead_zone * 0.6
                if offset_dist < soft_start:
                    response = 0.0
                elif offset_dist < dead_zone:
                    response = self._smoothstep(soft_start, dead_zone, offset_dist)
                else:
                    response = 1.0

                if response > 0:
                    effective_dist = offset_dist * response
                    target_cx = self.crop_cx + (offset_x / offset_dist) * effective_dist
                    target_cy = self.crop_cy + (offset_y / offset_dist) * effective_dist

                    # 方向阻尼：平移方向反转时增加阻尼防抖
                    pan_dx = target_cx - self.crop_cx
                    pan_dy = target_cy - self.crop_cy
                    dot_product = pan_dx * self.prev_pan_dx + pan_dy * self.prev_pan_dy
                    damp = self.direction_damp if dot_product < 0 else 1.0

                    eff_alpha = self.pan_alpha * damp
                    self.crop_cx = eff_alpha * target_cx + (1 - eff_alpha) * self.crop_cx
                    self.crop_cy = eff_alpha * target_cy + (1 - eff_alpha) * self.crop_cy

                    self.prev_pan_dx = pan_dx
                    self.prev_pan_dy = pan_dy

            # ================================================================
            # 3. 变焦控制（连续比例控制，替代二态滞回开关）
            # ================================================================
            current_pad_ratio = self.smooth_crop_size / self.smooth_face_size if self.smooth_face_size > 0 else self.default_pad_ratio
            ideal_pad_ratio = self.default_pad_ratio
            low_t = self.pad_ratio_range[0]
            high_t = self.pad_ratio_range[1]

            ratio_error = current_pad_ratio - ideal_pad_ratio

            if abs(ratio_error) > 0.01:
                ideal_crop_size = self.smooth_face_size * ideal_pad_ratio

                if current_pad_ratio < low_t:
                    urgency = self._smoothstep(ideal_pad_ratio, low_t, current_pad_ratio)
                    urgency = max(urgency, abs(ratio_error) * 0.5)
                elif current_pad_ratio > high_t:
                    urgency = self._smoothstep(high_t, ideal_pad_ratio + (ideal_pad_ratio - high_t), current_pad_ratio)
                    urgency = max(urgency, abs(ratio_error) * 0.5)
                else:
                    urgency = abs(ratio_error) * 2.0

                eff_zoom_alpha = self.zoom_alpha * min(urgency, 2.0)
                self.smooth_crop_size = eff_zoom_alpha * ideal_crop_size + (1 - eff_zoom_alpha) * self.smooth_crop_size

        # 画面边界约束：裁切框不超出画面
        crop_size = self.smooth_crop_size
        crop_size = min(crop_size, min(fh, fw))

        crop_cx = max(crop_size / 2.0, min(self.crop_cx, fw - crop_size / 2.0))
        crop_cy = max(crop_size / 2.0, min(self.crop_cy, fh - crop_size / 2.0))

        # 面部完整性约束：确保面部 bbox + 50px 余量整体在裁切框内
        face_margin = 50.0
        face_left = face_cx - face_w / 2.0 - face_margin
        face_right = face_cx + face_w / 2.0 + face_margin
        face_top = face_cy - face_h / 2.0 - face_margin
        face_bottom = face_cy + face_h / 2.0 + face_margin

        crop_x = crop_cx - crop_size / 2.0
        crop_y = crop_cy - crop_size / 2.0

        # 如果面部（含余量）超出裁切框，移动裁切框使面部完整包含
        if face_left < crop_x:
            crop_cx = face_left + crop_size / 2.0
        if face_right > crop_x + crop_size:
            crop_cx = face_right - crop_size / 2.0
        if face_top < crop_y:
            crop_cy = face_top + crop_size / 2.0
        if face_bottom > crop_y + crop_size:
            crop_cy = face_bottom - crop_size / 2.0

        # 面部完整性约束后再做画面边界约束（面部在画面外时以画面边界为准）
        crop_cx = max(crop_size / 2.0, min(crop_cx, fw - crop_size / 2.0))
        crop_cy = max(crop_size / 2.0, min(crop_cy, fh - crop_size / 2.0))

        crop_x = crop_cx - crop_size / 2.0
        crop_y = crop_cy - crop_size / 2.0

        actual_pad_ratio = crop_size / self.smooth_face_size if self.smooth_face_size > 0 else self.default_pad_ratio

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


def bounce_audio(wav_path, output_dir):
    """对音频做倒放拼接（正序+倒序），时长翻倍

    Args:
        wav_path: 原始 wav 文件路径
        output_dir: 输出目录

    Returns:
        拼接后的 wav 文件路径，失败时返回原始 wav_path
    """
    output_path = os.path.join(output_dir, 'aud.wav')

    # 用 ffmpeg filter_complex 一次性完成：输入同一个文件两次，
    # 第二路做 areverse，然后 concat 拼接
    filter_str = '[0:a]asplit=2[a1][a2];[a2]areverse[ar];[a1][ar]concat=n=2:v=0:a=1[outa]'
    temp_output = os.path.join(output_dir, 'aud_bounced.wav')
    cmd = [
        'ffmpeg', '-y', '-i', wav_path,
        '-filter_complex', filter_str,
        '-map', '[outa]',
        '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        temp_output,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f'[WARN] ffmpeg audio bounce failed: {result.stderr[:300]}')
            return wav_path
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f'[WARN] ffmpeg audio bounce error: {e}')
        return wav_path

    if not os.path.exists(temp_output) or os.path.getsize(temp_output) == 0:
        print('[WARN] Audio bounce produced no output, keeping original')
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return wav_path

    # 替换原始 aud.wav
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_output, output_path)

    print(f'[INFO] Audio bounce complete: {wav_path} -> {output_path}')
    return output_path


def pack_sequence_to_video(frames_dir, audio_path, output_dir, base_name, fps=25.0):
    """将序列帧+音频封包为 mp4 预览视频

    Args:
        frames_dir: 序列帧目录（full_body_img/）
        audio_path: 音频文件路径（aud.wav）
        output_dir: 输出目录
        base_name: 输出文件名（不含扩展名）
        fps: 帧率

    Returns:
        mp4 文件路径，失败时返回 None
    """
    output_path = os.path.join(output_dir, f'{base_name}.mp4')

    # 获取帧文件列表，确定命名模式（数字排序）
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                         key=lambda x: int(os.path.splitext(x)[0]))
    if not frame_files:
        print('[WARN] No frames found for preview video')
        return None

    frame_pattern = os.path.join(frames_dir, '%d.jpg')

    pack_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-start_number', '0',
        '-i', frame_pattern,
        '-i', audio_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'aac', '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        output_path,
    ]
    try:
        result = subprocess.run(pack_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f'[WARN] ffmpeg pack video failed: {result.stderr[:300]}')
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f'[WARN] ffmpeg pack video error: {e}')
        return None

    if not os.path.exists(output_path):
        print('[WARN] Preview video not generated')
        return None

    print(f'[INFO] Preview video: {output_path}')
    return output_path


def bounce_metadata(metadata):
    """对 metadata 帧列表做倒序拼接（正序 + 倒序）

    Args:
        metadata: 原始 metadata dict

    Returns:
        新的 metadata dict，帧数翻倍，后半段为前半段倒序
    """
    import copy
    result = copy.deepcopy(metadata)
    original_frames = metadata['frames']
    original_count = len(original_frames)

    # 倒序帧（深拷贝）
    reversed_frames = copy.deepcopy(original_frames[::-1])

    # 重新编号 frame_idx，添加 original_frame_idx
    for i, frame in enumerate(reversed_frames):
        frame['frame_idx'] = original_count + i
        frame['original_frame_idx'] = original_frames[original_count - 1 - i]['frame_idx']

    result['frames'] = original_frames + reversed_frames
    result['bounced'] = True
    result['original_frame_count'] = original_count
    result['frame_count'] = len(result['frames'])

    return result


def bounce_video(video_path, metadata, output_dir, base_name, output_size):
    """对裁切结果视频做倒放拼接（正序 + 倒序）

    Args:
        video_path: 裁切结果视频路径
        metadata: 原始 metadata dict
        output_dir: 输出目录
        base_name: 文件基础名
        output_size: 输出尺寸

    Returns:
        (bounced_video_path, bounced_metadata) 或 (原路径, 原metadata) 失败时
    """
    reverse_path = os.path.join(output_dir, f'{base_name}_head_{output_size}_reverse.mp4')
    concat_list_path = os.path.join(output_dir, f'{base_name}_concat_list.txt')
    bounced_path = os.path.join(output_dir, f'{base_name}_head_{output_size}_bounced.mp4')

    # Step 1: 生成倒放视频（含音频倒放）
    reverse_cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vf', 'reverse',
        '-af', 'areverse',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'aac', '-b:a', '192k',
        reverse_path
    ]
    try:
        result = subprocess.run(reverse_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f'[WARN] ffmpeg reverse failed: {result.stderr[:300]}')
            return video_path, metadata
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f'[WARN] ffmpeg reverse error: {e}')
        return video_path, metadata

    # Step 2: 创建 concat 列表文件
    with open(concat_list_path, 'w', encoding='utf-8') as f:
        f.write(f"file '{os.path.basename(video_path)}'\n")
        f.write(f"file '{os.path.basename(reverse_path)}'\n")

    # Step 3: concat 拼接
    concat_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_list_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'aac', '-b:a', '192k',
        bounced_path
    ]
    try:
        result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f'[WARN] ffmpeg concat failed: {result.stderr[:300]}')
            if os.path.exists(reverse_path):
                os.remove(reverse_path)
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
            return video_path, metadata
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f'[WARN] ffmpeg concat error: {e}')
        if os.path.exists(reverse_path):
            os.remove(reverse_path)
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
        return video_path, metadata

    # Step 4: 清理临时文件
    if os.path.exists(reverse_path):
        os.remove(reverse_path)
    if os.path.exists(concat_list_path):
        os.remove(concat_list_path)
    if os.path.exists(video_path):
        os.remove(video_path)

    # Step 5: 处理 metadata
    bounced_metadata = bounce_metadata(metadata)

    print(f'[INFO] Bounce complete: {len(bounced_metadata["frames"])} frames (original: {len(metadata["frames"])})')
    return bounced_path, bounced_metadata


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
                  conf_threshold=0.5,
                  preprocess=True, enhance=True, bounce=True,
                  gfpgan_model_path=None, enhance_strength=0.4,
                  base_name=None, no_bounce=False, no_enhance=False):
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]

    # 所有输出统一保存到 {output_dir}/{base_name}/ 子目录
    output_dir = os.path.join(output_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 前置重封装
    actual_video_path = video_path
    original_fps = None
    was_preprocessed = False
    if preprocess:
        actual_video_path, original_fps, was_preprocessed = preprocess_video(
            video_path, output_dir, base_name
        )

    metadata_path = os.path.join(output_dir, f'{base_name}_head_{output_size}_metadata.json')

    # Step 2: 拆帧和音频提取
    print('[INFO] Splitting video to frames...')
    frames_dir, frame_count = split_video_to_frames(actual_video_path, output_dir, base_name)
    if frame_count == 0:
        print('[ERROR] No frames extracted from video')
        return None, None

    audio_path = extract_audio_from_video(actual_video_path, output_dir)

    # 获取视频信息
    cap = cv2.VideoCapture(actual_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f'[INFO] Resolution: {frame_w}x{frame_h}, FPS: {fps:.2f}, Frames: {frame_count}')

    # Step 3: 初始化检测器、tracker、增强器
    detector = SCRFD(onnx_model_path, confThreshold=conf_threshold, nmsThreshold=0.5)
    tracker = HeadTracker(
        pan_alpha=pan_alpha, zoom_alpha=zoom_alpha,
        dead_zone_ratio=dead_zone_ratio, pad_ratio_range=pad_ratio_range,
        min_face_size=min_face_size, output_size=output_size,
    )

    do_enhance = enhance and not no_enhance
    enhancer = None
    if do_enhance:
        if gfpgan_model_path is None:
            gfpgan_model_path = os.path.join(os.path.dirname(__file__), 'model', 'gfpgan-1024.onnx')
        enhancer = CropFaceEnhancer(gfpgan_model_path, strength=enhance_strength)

    do_bounce = bounce and not no_bounce

    # Step 4: 逐帧处理 — 从 original_img/ 读取 → 检测 → 裁切 → 增强 → 写入 full_body_img/
    full_body_dir = os.path.join(output_dir, 'full_body_img')
    os.makedirs(full_body_dir, exist_ok=True)
    # original_img/ 已由 split_video_to_frames 创建

    metadata = {
        'source_video': video_path,
        'output_size': output_size,
        'fps': fps,
        'source_resolution': [frame_w, frame_h],
        'pan_alpha': pan_alpha,
        'zoom_alpha': zoom_alpha,
        'dead_zone_ratio': dead_zone_ratio,
        'pad_ratio_range': list(pad_ratio_range),
        'preprocessed': was_preprocessed,
        'enhanced': do_enhance and enhancer is not None and enhancer.session is not None,
        'enhance_strength': enhance_strength if do_enhance else 0.0,
        'split_frames': True,
        'output_mode': 'sequence_frames',
        'frames': []
    }

    # 获取排序后的帧文件列表（按文件名中的数字排序，避免字符串排序导致 10<2 的问题）
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                         key=lambda x: int(os.path.splitext(x)[0]))

    print('[INFO] Processing frames...')
    for frame_idx, fname in enumerate(tqdm(frame_files)):
        frame = cv2.imread(os.path.join(frames_dir, fname))
        if frame is None:
            print(f'[WARN] Cannot read frame: {fname}')
            continue

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

        # 面部增强（方案 D：局部面部增强）
        if enhancer is not None and enhancer.session is not None and face_result is not None:
            fx1, fy1, fw, fh, fconf = face_result
            crop_x = crop_info['crop_x']
            crop_y = crop_info['crop_y']
            face_bbox_in_crop = [
                (fx1 - crop_x) * scale_factor,
                (fy1 - crop_y) * scale_factor,
                fw * scale_factor,
                fh * scale_factor,
            ]
            resized = enhancer.enhance(resized, face_bbox=face_bbox_in_crop)

        # 直接输出为序列帧
        out_path = os.path.join(full_body_dir, f'{frame_idx}.jpg')
        cv2.imwrite(out_path, resized)

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

    # Step 5: 倒放拼接（帧文件追加倒序）
    final_metadata = metadata
    if do_bounce:
        # 追加倒序裁切帧到 full_body_dir（数字排序）
        existing_files = sorted([f for f in os.listdir(full_body_dir) if f.endswith('.jpg')],
                                key=lambda x: int(os.path.splitext(x)[0]))
        original_count = len(existing_files)
        for i, fname in enumerate(reversed(existing_files)):
            src = os.path.join(full_body_dir, fname)
            img = cv2.imread(src)
            dst_name = f'{original_count + i}.jpg'
            cv2.imwrite(os.path.join(full_body_dir, dst_name), img)

        # 同步追加倒序原帧到 original_img（即 frames_dir，数字排序）
        orig_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
                            key=lambda x: int(os.path.splitext(x)[0]))
        orig_count = len(orig_files)
        for i, fname in enumerate(reversed(orig_files)):
            src = os.path.join(frames_dir, fname)
            img = cv2.imread(src)
            dst_name = f'{orig_count + i}.jpg'
            cv2.imwrite(os.path.join(frames_dir, dst_name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        print(f'[INFO] Bounce: {original_count} → {original_count * 2} frames (full_body_img + original_img)')
        final_metadata = bounce_metadata(metadata)

        # 音频倒放拼接
        if audio_path is not None and os.path.exists(audio_path):
            audio_path = bounce_audio(audio_path, output_dir)

    # Step 6: 清理前置处理临时文件
    if was_preprocessed and os.path.exists(actual_video_path) and actual_video_path != video_path:
        os.remove(actual_video_path)
        print('[INFO] Preprocessed temporary file removed.')

    # Step 7: 保存 metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(final_metadata, f, indent=2, ensure_ascii=False)

    # Step 8: 生成预览视频
    if audio_path is not None and os.path.exists(audio_path):
        pack_sequence_to_video(full_body_dir, audio_path, output_dir, base_name, fps)

    print(f'[INFO] Output: {full_body_dir}')
    print(f'[INFO] Original: {frames_dir}')
    print(f'[INFO] Metadata: {metadata_path}')
    print(f'[INFO] Total frames processed: {len(final_metadata["frames"])}')

    return final_metadata, metadata_path


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
    parser.add_argument('--no_preprocess', action='store_true',
                        help='Skip video preprocessing (remux to 25fps)')
    parser.add_argument('--no_enhance', action='store_true',
                        help='Skip GFPGAN face enhancement')
    parser.add_argument('--no_bounce', action='store_true',
                        help='Skip bounce (reverse concatenation)')
    parser.add_argument('--gfpgan_model', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'model', 'gfpgan-1024.onnx'),
                        help='Path to GFPGAN ONNX model')
    parser.add_argument('--enhance_strength', type=float, default=0.4,
                        help='Face enhancement strength (0.0~1.0, default: 0.4)')

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
        preprocess=not opt.no_preprocess,
        enhance=not opt.no_enhance,
        bounce=not opt.no_bounce,
        no_bounce=opt.no_bounce,
        no_enhance=opt.no_enhance,
        gfpgan_model_path=opt.gfpgan_model,
        enhance_strength=opt.enhance_strength,
    )
