"""Check an extracted FF++ crop tree against what train.py / eval.py / dataset.py expect.

usage: python verify_ffpp.py /kaggle/temp/FFPP/c23 [--train_real 20 --train_fake 5 --eval_frames 20]

Expected layout: <root>/{train,valid,test}/{original,Deepfakes,Face2Face,FaceSwap,NueralTextures}/<video>/<k>.png
Official FF++ split sizes: 720 / 140 / 140 videos per method (1000 videos = 500 pairs).
"""
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('--train_real', type=int, default=20, help='crops per REAL train video')
ap.add_argument('--train_fake', type=int, default=5, help='crops per FAKE train video')
ap.add_argument('--eval_frames', type=int, default=20, help='crops per valid/test video')
a = ap.parse_args()

METHODS = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NueralTextures']
N = {'train': 720, 'valid': 140, 'test': 140}

bad = 0
print(f"{'split':6} {'method':15} {'videos':>7} {'expect':>7} {'few_crops':>10} {'empty':>6} {'stray':>6}")
for sp in N:
    for m in METHODS:
        d = os.path.join(a.root, sp, m)
        need = a.eval_frames if sp != 'train' else (a.train_real if m == 'original' else a.train_fake)
        vids = sorted(os.listdir(d)) if os.path.isdir(d) else []
        few = empty = stray = 0
        for v in vids:
            files = os.listdir(os.path.join(d, v))
            ok = [f for f in files if f.endswith('.png') and f[:-4].isdigit()]
            stray += len(files) - len(ok)      # dataset.py does int(name[:-4]) on every file
            empty += (len(ok) == 0)            # empty folder -> ZeroDivisionError in dataset.py
            few += (len(ok) < need)            # fewer crops than wanted (dataset.py would repeat crops)
        flag = '' if (len(vids) == N[sp] and few == 0 and empty == 0 and stray == 0) else '   <-- check'
        bad += bool(flag)
        print(f"{sp:6} {m:15} {len(vids):7d} {N[sp]:7d} {few:10d} {empty:6d} {stray:6d}{flag}")
print('ALL OK' if bad == 0 else f'{bad} row(s) need attention (see extract_log.csv for the video ids)')
