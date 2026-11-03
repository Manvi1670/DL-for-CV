## -*- coding: utf-8 -*-
import os, sys
sys.setrecursionlimit(15000)
import torch
import numpy as np
import random
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import time
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm
from dataset import FFPP_Dataset,TestDataset
import timm
from utils import *
from ViT_MoE import *



def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def test(args, model,test_loader,model_path):

    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(torch.cuda.current_device()))
    # accept both the per-epoch .tar (dict with 'model_state_dict') and the best-model .pkl (raw state dict)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    print('start test mode...')
    model.eval()
    video_predictions=[]
    video_labels=[]
    frame_predictions=[]
    frame_labels=[]
    with torch.no_grad():
        st_time = time.time()

        for inputs,labels in tqdm(test_loader,total=len(test_loader),ncols=70,leave=False,unit='step'):
            inputs = inputs.cuda()
            inputs = inputs.squeeze(0)
            labels = labels.cuda()

            outputs,_= model(inputs)

            outputs = F.softmax(outputs, dim=-1)
            frame = outputs.shape[0]
            frame_predictions.extend(outputs[:,1].cpu().tolist())
            frame_labels.extend(labels.expand(frame).cpu().tolist())
            pre = torch.mean(outputs[:,1])
            video_predictions.append(pre.cpu().item())
            video_labels.append(labels.cpu().item())
        period = time.time() - st_time
    print('total time: {}s'.format(period))

    frame_results = cal_metrics(frame_labels, frame_predictions, threshold=0.5)
    video_results = cal_metrics(video_labels, video_predictions, threshold=0.5)  # 'best' 'auto' or float
    print('Test result: V_Acc: {:.2%}, V_Auc: {:.4} V_EER:{:.2%} F_Acc: {:.2%}, F_Auc: {:.4} F_EER:{:.2%}'
             .format(video_results.ACC, video_results.AUC, video_results.EER,frame_results.ACC, frame_results.AUC, frame_results.EER))

    # added: machine-readable result so numbers in the report can be traced to a file
    if args.out_json:
        import json
        res = {'model_path': model_path, 'dataset': args.dataset, 'test_root': args.test_root,
               'frames_per_video': args.frames, 'n_videos': len(video_labels),
               'video': {'AUC': float(video_results.AUC), 'EER': float(video_results.EER), 'ACC': float(video_results.ACC)},
               'frame': {'AUC': float(frame_results.AUC), 'EER': float(frame_results.EER), 'ACC': float(frame_results.ACC)}}
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device','-dv', type=int, default=0, help="specify which GPU to use")
    parser.add_argument('--model_path', '-md', type=str, default='models/train/models_params_0.tar')
    parser.add_argument('--resume','-rs', type=int, default=-1, help="which epoch continue to train")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--record_step', type=int, default=100, help="the iteration number to record train state")

    parser.add_argument('--batch_size','-bs', type=int, default=32)
    parser.add_argument('--learning_rate','-lr', type=float, default=1e-3)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    # --- added for Kaggle / reproducibility (original hardcoded: Celeb_DF test path, dataset='CDF', frame=100) ---
    parser.add_argument('--test_root', type=str, required=True,
                        help="folder of the test set (FF++: method/video/frames; CDF/DFDC_P: real*/fake* folders of video/frames)")
    parser.add_argument('--dataset', type=str, default='CDF', choices=['FFPP', 'CDF', 'DFDC_P'])
    parser.add_argument('--frames', type=int, default=100, help="frames sampled per video (original CDF setting: 100)")
    parser.add_argument('--out_json', type=str, default='', help="optional path to write the metrics as JSON")
    args = parser.parse_args()


    start_time = time.time()
    setup_seed(2024)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.device)
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    test_dataset = TestDataset(args.test_root, dataset=args.dataset, frame=args.frames)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


    model = vit_base_patch16_224_in21k(pretrained=True,num_classes=2)
    model = model.cuda()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    print('Start eval process...')
    test(args, model,test_loader,args.model_path)
    duration = time.time()-start_time
    print('The run time is {}h {}m'.format(int(duration//3600),int(duration%3600//60)))
