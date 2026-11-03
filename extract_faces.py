#!/usr/bin/env python
"""Extract face crops from FaceForensics++ videos into the layout dataset.py expects:

  <out_root>/{train,valid,test}/{original,Deepfakes,Face2Face,FaceSwap,NueralTextures}/<video_id>/<k>.png

Rules imposed by dataset.py (do not break them):
  * frame files are named <integer>.<3-letter ext>  (it does int(name[:-4]))
  * no other files/folders inside the split or video folders
  * no spaces in any path
  * dataset.py samples `frame` images per video with interval = len//frame, so extract
    exactly `frame` crops per video (default 20) to cover the whole video.

Usage (smoke test on whatever videos are on disk):
  python extract_faces.py --ff_root /kaggle/temp/FF --out_root /kaggle/working/smoke --smoke 3
Usage (real run, official splits):
  python extract_faces.py --ff_root /kaggle/temp/FF --splits_dir splits \
      --out_root /kaggle/working/FaceForensics++/c23 --frames 20
"""
import argparse, csv, glob, json, os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

# source folder name -> output folder name (spelling matches the authors' layout)
METHODS = {'original': 'original', 'Deepfakes': 'Deepfakes', 'Face2Face': 'Face2Face',
           'FaceSwap': 'FaceSwap', 'NeuralTextures': 'NueralTextures'}
SPLIT_FILE = {'train': 'train.json', 'valid': 'val.json', 'test': 'test.json'}


def video_dir(ff_root, method, comp):
    if method == 'original':
        return os.path.join(ff_root, 'original_sequences', 'youtube', comp, 'videos')
    return os.path.join(ff_root, 'manipulated_sequences', method, comp, 'videos')


def crop_face(img, box, scale, size):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * scale / 2
    h, w = img.shape[:2]
    a, b = int(max(0, cx - s)), int(max(0, cy - s))
    c, d = int(min(w, cx + s)), int(min(h, cy + s))
    crop = img[b:d, a:c]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def process_video(vpath, outdir, n_keep, mtcnn, a):
    """Returns (total_frames_in_video, crops_saved)."""
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return 0, 0
    n_cand = min(total, int(np.ceil(n_keep * a.oversample)))
    wanted = set(np.unique(np.linspace(0, total - 1, n_cand).round().astype(int)).tolist())
    frames = []
    for i in range(total):                       # sequential read: exact frame indices
        if not cap.grab():
            break
        if i in wanted:
            ok, f = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    crops = []
    for k in range(0, len(frames), a.det_batch):
        chunk = frames[k:k + a.det_batch]
        boxes, _ = mtcnn.detect([Image.fromarray(f) for f in chunk])
        for f, b in zip(chunk, boxes):
            if b is None:
                continue
            b = np.asarray(b)
            areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            c = crop_face(f, b[areas.argmax()], a.scale, a.size)   # largest face
            if c is not None:
                crops.append(c)
    if len(crops) > n_keep:                      # keep n_keep, evenly spread
        sel = np.unique(np.linspace(0, len(crops) - 1, n_keep).round().astype(int))
        crops = [crops[i] for i in sel]

    os.makedirs(outdir, exist_ok=True)
    for j, c in enumerate(crops):
        p = os.path.join(outdir, f'{j}.{a.fmt}')
        cv2.imwrite(p, cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
    return total, len(crops)


def build_jobs(a):
    jobs = []  # (split, method, video_id)
    if a.smoke:
        for m in a.methods:
            files = sorted(glob.glob(os.path.join(video_dir(a.ff_root, m, a.comp), '*.mp4')))[:a.smoke]
            jobs += [('smoke', m, os.path.splitext(os.path.basename(f))[0]) for f in files]
        return jobs
    for split in a.splits:
        pairs = json.load(open(os.path.join(a.splits_dir, SPLIT_FILE[split])))
        if a.limit:
            pairs = pairs[:a.limit]
        for m in a.methods:
            for x, y in pairs:
                ids = [x, y] if m == 'original' else [f'{x}_{y}', f'{y}_{x}']
                jobs += [(split, m, v) for v in ids]
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ff_root', required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--splits_dir', default='splits')
    p.add_argument('--comp', default='c23')
    p.add_argument('--frames', type=int, default=20, help='crops per video')
    p.add_argument('--real_train_frames', type=int, default=0,
                   help='if >0, crops per REAL train video (e.g. 80 = unique frames instead of 4x duplicates)')
    p.add_argument('--oversample', type=float, default=1.5, help='candidate frames = frames*oversample')
    p.add_argument('--scale', type=float, default=1.3, help='crop box scale (our choice; keep identical for ALL datasets)')
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--fmt', default='png', choices=['png', 'jpg'])
    p.add_argument('--det_batch', type=int, default=8)
    p.add_argument('--methods', nargs='+', default=list(METHODS))
    p.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    p.add_argument('--limit', type=int, default=0, help='first N pairs per split (quick partial run)')
    p.add_argument('--smoke', type=int, default=0, help='ignore splits; first N videos on disk per method')
    a = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, device=dev)
    jobs = build_jobs(a)
    print(f'{len(jobs)} videos, device={dev}, frames={a.frames}, scale={a.scale}, fmt={a.fmt}')

    os.makedirs(a.out_root, exist_ok=True)
    log_path = os.path.join(os.path.dirname(a.out_root.rstrip('/')) or '.', 'extract_log.csv')
    new_log = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as lf:
        w = csv.writer(lf)
        if new_log:
            w.writerow(['split', 'method', 'video', 'total_frames', 'crops_saved'])
        for split, m, v in tqdm(jobs):
            n_keep = a.real_train_frames if (m == 'original' and split == 'train' and a.real_train_frames) else a.frames
            outdir = os.path.join(a.out_root, split, METHODS[m], v)
            if os.path.isdir(outdir) and len(os.listdir(outdir)) >= n_keep:
                continue                                          # resume: already done
            vpath = os.path.join(video_dir(a.ff_root, m, a.comp), v + '.mp4')
            if not os.path.exists(vpath):
                w.writerow([split, m, v, -1, 0])                  # missing video
                continue
            total, kept = process_video(vpath, outdir, n_keep, mtcnn, a)
            w.writerow([split, m, v, total, kept])
            lf.flush()


if __name__ == '__main__':
    main()
