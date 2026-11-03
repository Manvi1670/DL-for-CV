"""Does the cross-entropy loss reach the MoE gates?

Run twice, in separate processes (ST_GATE is read when ViT_MoE is imported):
    ST_GATE=0 python grad_check.py      # baseline: expect gate grads None / 0
    ST_GATE=1 python grad_check.py      # ours: expect gate grads non-zero

Why the perturbation: every expert is zero-initialised (adapter_up, Lora_b), so E(x)=0 at step 0 and the
straight-through gradient c = <dL/dy, E(x)> is exactly 0 for a fresh model. We give those weights small
random values so the check reflects a model after a few optimiser steps. CE loss only (no moe_loss).
"""
import torch, torch.nn as nn
import ViT_MoE
from ViT_MoE import vit_base_patch16_224_in21k

torch.manual_seed(0)
m = vit_base_patch16_224_in21k(pretrained=True, num_classes=2).cuda().train()
with torch.no_grad():
    for n, p in m.named_parameters():
        if 'adapter_up' in n or 'Lora_b' in n:
            p.normal_(0, 0.02)

x = torch.randn(16, 3, 224, 224).cuda()
y = torch.randint(0, 2, (16,)).cuda()
out, _moe = m(x)
nn.CrossEntropyLoss()(out, y).backward()

tot, none, count = {'w_gate': 0.0, 'w_noise': 0.0}, 0, 0
for n, p in m.named_parameters():
    for k in tot:
        if k in n:
            count += 1
            if p.grad is None:
                none += 1
            else:
                tot[k] += p.grad.abs().sum().item()
print('ST_GATE =', getattr(ViT_MoE, 'ST_GATE', 'n/a (unpatched file)'))
print(f'gate tensors: {count}, with grad None: {none}')
print('sum |grad|:', tot)
