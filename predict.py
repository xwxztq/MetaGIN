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
parser.add_argument('--train',  type=int,  default=12,        help='number of training periods')
parser.add_argument('--save',   type=str,  default=None,      help='dir to save checkpoints')
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
batch_size, cos_period, num_period = 256, 12, args.train
print('#torch:', pt.__version__, pt.version.cuda)

from data import dataset, dataidx, dataeval
train_loader = DataLoader(dataset[dataidx["train"]], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
valid_loader = DataLoader(dataset[dataidx["valid"]], batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
test_loader  = DataLoader(dataset[dataidx["test-dev"]], batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
print('#loader:', batch_size, len(train_loader))

model = MetaGIN(**model_config).cuda()
param = get_param(model, lr_base, wd_base, lr_min)
optim = Adan(param, lr_base, weight_decay=wd_base)
sched = Scheduler(optim, cos_period*2, cos_period)
print('#optim:', '%.2e'%lr_base, '%.2e'%wd_base, cos_period, num_period)


loss_fn = pt.nn.L1Loss()
def train(model, loader, optim, param):
    model.train()
    loss_accum = 0

    optim.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.cuda()

        pred = model(batch)
        loss = loss_fn(pred.view(-1,), batch.y)
        loss.backward()
        optim.step()
        clamp_param(param)

        loss_accum += loss.detach().cpu().item()
        optim.zero_grad()

    return loss_accum / (step + 1)

def eval(model, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.cuda()

        with pt.no_grad():
            pred = model(batch)
            y_true.append(batch.y.detach().cpu())
            y_pred.append(pred.view(-1,).detach().cpu())

    y_true = pt.cat(y_true, dim = 0)
    y_pred = pt.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

def test(model, statefn, loader):
    model_copy = deepcopy(model).cuda()
    model_copy.load_state_dict(pt.load(statefn))
    model_copy.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.cuda()

        with pt.no_grad():
            pred = model_copy(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = pt.cat(y_pred, dim = 0)

    return y_pred.cpu().detach().numpy()


print(); print('#training...')
best_mae, best_epoch, t0 = 9999, -1, time()
for epoch in range(cos_period*num_period):
    epoch_lr, epoch_wd = sched.step(epoch, 1e-7)
    train_mae = train(model, train_loader, optim, param)
    valid_mae = eval(model, valid_loader, dataeval)
    eta = (time() - t0) / (epoch + 1) * (cos_period * num_period - epoch - 1) / 3600

    if valid_mae < best_mae:
        best_mae, best_epoch = valid_mae, epoch
        print('#epoch[%d.%02d]:  %.4f %.4f  %.2e %.2e  %.1fh *' \
             % (epoch//cos_period, epoch%cos_period, train_mae, valid_mae, epoch_lr, epoch_wd, eta))
    else:
        print('#epoch[%d.%02d]:  %.4f %.4f  %.2e %.2e  %.1fh' \
             % (epoch//cos_period, epoch%cos_period, train_mae, valid_mae, epoch_lr, epoch_wd, eta))

    if args.save is not None:
        pt.save(model.state_dict(), args.save + '/model%04d.pt' % epoch)
if args.save is not None:
    test_pred = test(model, args.save + '/model%04d.pt' % best_epoch, test_loader)
    dataeval.save_test_submission({'y_pred': test_pred}, args.save, mode='test-dev')
    print('#saved[%d]: test-dev' % best_epoch)
print('#done!!!')

