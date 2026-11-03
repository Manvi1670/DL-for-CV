#!/usr/bin/env bash
# FaceForensics++ (c23): download -> MTCNN face crops -> zip per split.
# Run on a Kaggle GPU notebook with Internet ON.
#
#   bash scripts/prep_ffpp.sh /path/to/your_FF_download_script.py
#
# Raw videos and crops go to /kaggle/temp (large), zips and the log go to /kaggle/working (20 GB limit).
# Frozen data plan: train = 720 videos/method, 5 crops per FAKE video, 20 crops per REAL video;
#                   valid and test = 140 videos/method, 20 crops per video.  (train.py is run with --train_frames 5)
set -euo pipefail

DL="${1:?usage: bash scripts/prep_ffpp.sh <your FF++ download script .py>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FF=/kaggle/temp/FF
OUT=/kaggle/temp/FFPP/c23
SPLITS=/kaggle/working/splits
mkdir -p "$FF" "$OUT" "$SPLITS"

echo "== 1/5 official split lists"
for s in train val test; do
  [ -s "$SPLITS/$s.json" ] || curl -sSfL -o "$SPLITS/$s.json" \
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/$s.json"
done
python3 -I - <<'EOF'
import json
n = {s: len(json.load(open(f'/kaggle/working/splits/{s}.json'))) for s in ('train', 'val', 'test')}
print('pairs per split:', n)
assert sum(n.values()) == 500, n          # 1000 videos = 500 pairs
EOF

echo "== 2/5 download all 1000 videos per method (c23). Flags below are the official script's; use the ones that worked for your sample"
for d in original Deepfakes Face2Face FaceSwap NeuralTextures; do
  echo | python3 "$DL" "$FF" -d "$d" -c c23 -t videos
done
du -sh "$FF"

echo "== 3/5 face crops: train (5 per fake video, 20 per real video)"
python3 "$REPO/extract_faces.py" --ff_root "$FF" --splits_dir "$SPLITS" --out_root "$OUT" \
  --splits train --frames 5 --real_train_frames 20

echo "== 4/5 face crops: valid and test (20 per video)"
python3 "$REPO/extract_faces.py" --ff_root "$FF" --splits_dir "$SPLITS" --out_root "$OUT" \
  --splits valid test --frames 20

echo "== 5/5 verify, then zip (PNG does not compress, so zips are about the size of the crops)"
python3 "$REPO/scripts/verify_ffpp.py" "$OUT" | tee /kaggle/working/verify_ffpp.txt
cp /kaggle/temp/FFPP/extract_log.csv /kaggle/working/
python3 -I - <<'EOF'
import shutil
for s in ('train', 'valid', 'test'):
    shutil.make_archive(f'/kaggle/working/ffpp_c23_{s}', 'zip', root_dir='/kaggle/temp/FFPP/c23', base_dir=s)
    print('zipped', s)
EOF
ls -lh /kaggle/working/*.zip
df -h /kaggle/working
