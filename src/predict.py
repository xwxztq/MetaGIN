import re
import argparse
from time import time
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch as pt
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from torch_geometric.loader import DataLoader

from adan import Adan
from model import MetaGIN
from optim import get_param, clamp_param, Scheduler


parser = argparse.ArgumentParser(description='CoAtGIN for graph-based molecule property prediction')
parser.add_argument('--model',  type=str,  default='base31',  help='base31, base32, deep31 or wide31 (default: base31)')
parser.add_argument('--checkpoint',  type=str,  required=True,  help='checkpoint file to load')
parser.add_argument('--smiles_file',  type=str,  required=True,  help='smiles file to load')
args = parser.parse_args()

if args.model.startswith('base'):
    model_config = {'width': 256, 'nhead':  8, 'depth': 4, 'kernel': [1, 1, 1, 1]}
elif args.model.startswith('deep'):
    model_config = {'width': 256, 'nhead':  8, 'depth': 4, 'kernel': [1, 2, 4, 1]}
elif args.model.startswith('wide'):
    model_config = {'width': 256, 'nhead': 16, 'depth': 4, 'kernel': [1, 1, 1, 1]}
else:
    assert False, args.model
model_name = re.sub('\D+', '', args.model)
assert len(model_name)==2 and 1<=int(model_name[0])<=3 and 1<=int(model_name[1]), args.model
for i in range(model_config['depth']):
    model_config['kernel'][i] *= int(model_name[1])
model_config['hop'] = int(model_name[0])


lr_base, lr_min, wd_base = 1e-3, 4e-5, 2e-2
batch_size = 256
print('#torch:', pt.__version__, pt.version.cuda)

from data import SmilesPCQM4Mv2Dataset, hetero_transform, cast_transform
smiles_list = np.loadtxt(args.smiles_file, dtype=str)

dataset = SmilesPCQM4Mv2Dataset(smiles_list,  pre_transform=hetero_transform, transform=cast_transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
print('#loader:', batch_size, len(loader))

model = MetaGIN(**model_config).cuda()
model.load_state_dict(pt.load(args.checkpoint))

model.eval()
y_pred = []

for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    batch = batch.cuda()

    with pt.no_grad():
        pred = model(batch)
        y_pred.append(pred.view(-1,).detach().cpu())

y_pred = pt.cat(y_pred, dim = 0)

print('#y_pred shape:', y_pred.shape)
print('#y_pred:', y_pred[:10])

print('#done!!!')

