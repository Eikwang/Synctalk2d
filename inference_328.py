import argparse
import os
import json
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet_328 import Model
from tqdm import tqdm
from utils import AudioEncoder, AudDataset, get_audio_features

import time
parser = argparse.ArgumentParser(description='Inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="ave")
parser.add_argument('--name', type=str, default="May")
parser.add_argument('--audio_path', type=str, default="demo/talk_hb.wav")
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--parsing', type=bool, default=False)
parser.add_argument('--original_video', type=str, default=None,
                    help='Path to original full-resolution video. If provided, output full-body video instead of cropped.')
parser.add_argument('--metadata', type=str, default=None,
                    help='Path to crop metadata JSON file (required with --original_video)')
args = parser.parse_args()

use_original = args.original_video is not None

if use_original and args.metadata is None:
    raise ValueError("--metadata is required when using --original_video")

checkpoint_path = os.path.join("./checkpoint", args.name)
checkpoint = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path), key=lambda x: int(x.split(".")[0]))[-1])
print(checkpoint)
save_path = os.path.join("./result", args.name+"_"+os.path.basename(args.audio_path).split(".")[0]+"_"+os.path.basename(checkpoint).split(".")[0]+".mp4")
dataset_dir = os.path.join("./dataset", args.name)
audio_path = args.audio_path
mode = args.asr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioEncoder().to(device).eval()
ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth', weights_only=False)
model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
dataset = AudDataset(audio_path)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
outputs = []
for mel in data_loader:
    mel = mel.to(device)
    with torch.no_grad():
        out = model(mel)
    outputs.append(out)
outputs = torch.cat(outputs, dim=0).cpu()
first_frame, last_frame = outputs[:1], outputs[-1:]
audio_feats = torch.cat([first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)],
                            dim=0).numpy()

lms_dir = os.path.join(dataset_dir, "landmarks/")

if use_original:
    cap_orig = cv2.VideoCapture(args.original_video)
    if not cap_orig.isOpened():
        raise ValueError("Cannot open original video: %s" % args.original_video)

    orig_w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_total = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap_orig.get(cv2.CAP_PROP_FPS)

    with open(args.metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    meta_frames = metadata['frames']

    len_img = min(orig_total, len(meta_frames)) - 1
    out_w, out_h = orig_w, orig_h
    out_fps = 25.0 if mode in ("hubert", "ave") else 20.0

    print('[INFO] Original video: %dx%d, %.2ffps, %d frames' % (orig_w, orig_h, orig_fps, orig_total))
    print('[INFO] Metadata frames: %d' % len(meta_frames))
    print('[INFO] Output: full-body video at original resolution')
else:
    img_dir = os.path.join(dataset_dir, "full_body_img/")
    len_img = len(os.listdir(img_dir)) - 1
    exm_img = cv2.imread(img_dir+"0.jpg")
    out_h, out_w = exm_img.shape[:2]
    out_fps = 25.0 if mode in ("hubert", "ave") else 20.0

if args.parsing:
    parsing_dir = os.path.join(dataset_dir, "parsing/")

video_writer = cv2.VideoWriter(save_path.replace(".mp4", "temp.mp4"),
                               cv2.VideoWriter_fourcc('M','J','P', 'G'),
                               out_fps, (out_w, out_h))

step_stride = 0
img_idx = 0

net = Model(6, mode).cuda()
net.load_state_dict(torch.load(checkpoint, weights_only=False))
net.eval()

for i in tqdm(range(audio_feats.shape[0])):
    if img_idx > len_img - 1:
        step_stride = -1
    if img_idx < 1:
        step_stride = 1
    img_idx += step_stride

    frame_idx = img_idx + args.start_frame

    if use_original:
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap_orig.read()
        if not ret:
            print('[WARN] Cannot read frame %d from original video' % frame_idx)
            continue
    else:
        img_path = img_dir + str(frame_idx) + '.jpg'
        img = cv2.imread(img_path)

    if args.parsing:
        parsing_path = parsing_dir + str(frame_idx) + '.png'
        parsing = cv2.imread(parsing_path)

    lms_path = lms_dir + str(frame_idx) + '.lms'

    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.float32)

    if use_original:
        if frame_idx < len(meta_frames):
            fm = meta_frames[frame_idx]
            crop_x = fm['crop_x']
            crop_y = fm['crop_y']
            crop_size = fm['crop_size']
            output_size = metadata.get('output_size', 512)
            scale = crop_size / float(output_size)
            lms[:, 0] = lms[:, 0] * scale + crop_x
            lms[:, 1] = lms[:, 1] * scale + crop_y

    lms = lms.astype(np.int32)

    xmin = lms[1][0]
    ymin = lms[52][1]
    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width

    oh, ow = img.shape[:2]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(ow, xmax)
    ymax = min(oh, ymax)

    crop_img = img[ymin:ymax, xmin:xmax]
    crop_img_par = crop_img.copy()
    if args.parsing:
        crop_parsing_img = parsing[ymin:ymax, xmin:xmax]

    ch, cw = crop_img.shape[:2]
    if cw <= 0 or ch <= 0:
        video_writer.write(img)
        continue

    crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:324, 4:324].copy()
    img_real_ex_ori = img_real_ex.copy()
    img_masked = cv2.rectangle(img_real_ex_ori, (5, 5, 310, 305), (0, 0, 0), -1)
    img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
    img_masked_T = torch.from_numpy(img_masked / 255.0)
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]

    audio_feat = get_audio_features(audio_feats, i)
    if mode == "hubert":
        audio_feat = audio_feat.reshape(32, 32, 32)
    if mode == "wenet":
        audio_feat = audio_feat.reshape(256, 16, 32)
    if mode == "ave":
        audio_feat = audio_feat.reshape(32, 16, 16)
    audio_feat = audio_feat[None]
    audio_feat = audio_feat.cuda()
    img_concat_T = img_concat_T.cuda()

    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]

    pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
    pred = np.array(pred, dtype=np.uint8)

    crop_img_ori[4:324, 4:324] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (cw, ch), interpolation=cv2.INTER_CUBIC)
    if args.parsing:
        parsing_mask = (crop_parsing_img == [0, 0, 255]).all(axis=2) | (crop_parsing_img == [255, 255, 255]).all(axis=2)
        crop_img_ori[parsing_mask] = crop_img_par[parsing_mask]
    img[ymin:ymax, xmin:xmax] = crop_img_ori
    video_writer.write(img)

if use_original:
    cap_orig.release()

video_writer.release()

os.system(f"ffmpeg -i {save_path.replace('.mp4', 'temp.mp4')} -i {audio_path} -c:v libx264 -c:a aac -crf 18 {save_path} -y")
if os.path.exists(save_path.replace('.mp4', 'temp.mp4')):
    os.remove(save_path.replace('.mp4', 'temp.mp4'))
print(f"[INFO] ===== save video to {save_path} =====")
