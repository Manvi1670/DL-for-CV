"""Evaluate one checkpoint on FF++ test / CDF / DFDC-P and append rows to a CSV.

CSV columns: model, dataset, level (frame/video), AUC, EER, ACC
Accepts both checkpoint formats train.py writes:
  models_params_<epoch>.tar          (dict with 'model_state_dict')
  model_params_best_*.pkl            (bare state_dict)

Example:
  python eval_csv.py --ckpt runs/baseline/model_params_best_xxx.pkl --tag baseline --out results.csv \
      --ffpp /path/c23/test --cdf /path/CDF/test --dfdcp /path/DFDCP
Only the datasets whose path you pass are evaluated.
"""
import argparse, csv, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import TestDataset
from utils import cal_metrics
from ViT_MoE import vit_base_patch16_224_in21k


def load_ckpt(model, path):
    ck = torch.load(path, map_location='cuda:0')
    sd = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
    model.load_state_dict(sd)


@torch.no_grad()
def run(model, loader):
    vp, vl, fp, fl = [], [], [], []
    for inputs, labels in loader:
        inputs = inputs.cuda().squeeze(0)
        labels = labels.cuda()
        out, _ = model(inputs)
        prob = F.softmax(out, dim=-1)[:, 1]
        fp.extend(prob.cpu().tolist())
        fl.extend(labels.expand(prob.shape[0]).cpu().tolist())
        vp.append(prob.mean().cpu().item())
        vl.append(labels.cpu().item())
    return cal_metrics(fl, fp, threshold=0.5), cal_metrics(vl, vp, threshold=0.5), len(vl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', required=True, help="value for the 'model' column, e.g. baseline / ours")
    p.add_argument('--out', default='results.csv')
    p.add_argument('--ffpp'); p.add_argument('--cdf'); p.add_argument('--dfdcp')
    p.add_argument('--frames_ffpp', type=int, default=20)
    p.add_argument('--frames_cdf', type=int, default=100, help='eval.py of the authors uses 100 for CDF')
    p.add_argument('--frames_dfdcp', type=int, default=20)
    p.add_argument('--num_workers', type=int, default=2)
    a = p.parse_args()

    model = vit_base_patch16_224_in21k(pretrained=True, num_classes=2).cuda()
    load_ckpt(model, a.ckpt)
    model.eval()

    jobs = [(n, k, path, fr) for n, k, path, fr in
            [('FFPP', 'FFPP', a.ffpp, a.frames_ffpp), ('CDF', 'CDF', a.cdf, a.frames_cdf),
             ('DFDC_P', 'DFDC_P', a.dfdcp, a.frames_dfdcp)] if path]
    new = not os.path.exists(a.out)
    with open(a.out, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['model', 'dataset', 'level', 'AUC', 'EER', 'ACC'])
        for name, kind, path, frames in jobs:
            ds = TestDataset(path, dataset=kind, frame=frames)
            loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.num_workers)
            t0 = time.time()
            fr, vr, nv = run(model, loader)
            print(f'{name}: {nv} videos, {frames} frames/video, {time.time()-t0:.0f}s')
            for level, r in (('frame', fr), ('video', vr)):
                w.writerow([a.tag, name, level, f'{r.AUC*100:.2f}', f'{r.EER*100:.2f}', f'{r.ACC*100:.2f}'])
                print(f'  {level}: AUC {r.AUC*100:.2f}  EER {r.EER*100:.2f}  ACC {r.ACC*100:.2f}')
            f.flush()


if __name__ == '__main__':
    main()
