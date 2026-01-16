import os
import os
import glob
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plts
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from torch.utils.tensorboard import SummaryWriter

from losses import *
from utils import *
from dataset import SurvivalDataset
from models import LinearProbingModel, LinearProbingSimple

from sksurv.metrics import concordance_index_censored


def process_surv(logits, label, censorship, loss_fn = None):
    results_dict = {'logits': logits}
    log_dict = {}
    
    if loss_fn is not None and label is not None:
        if isinstance(loss_fn, NLLSurvLoss):
            surv_loss_dict = loss_fn(logits = logits, times = label, censorships = censorship)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim = 1)
            risk = -torch.sum(survival, dim = 1).unsqueeze(dim = 1)
            results_dict.update({
                'hazards': hazards,
                'survival': survival,
                'risk': risk
            })
        
        elif isinstance(loss_fn, CoxLoss):
            # logits is log risk
            surv_loss_dict = loss_fn(logits=logits, times=label, censorships = censorship)
            risk = torch.exp(logits)
            results_dict['risk'] = risk
            
        elif isinstance(loss_fn, SurvRankingLoss):
            surv_loss_dict = loss_fn(z = logits, times = label, censorships = censorship)
            
        loss = surv_loss_dict['loss']
        log_dict['surv_loss'] = surv_loss_dict['loss'].item()
        log_dict.update({
            k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)
        })
        results_dict.update({'loss': loss})
        return results_dict, log_dict

def train_step(model, loader, optimizer, lr_scheduler, loss_fn, in_dropout = 0.0, print_every = 100, accum_steps = 32, args = None):
    model.train()
    
    meters = {}
    
    all_risk_scores, all_censorships, all_event_times = [], [], []
    
    for batch_id, batch in enumerate(loader):
        data = batch['img'].cuda()
        #(data.shape)
        if len(data.shape) > 3:
            data = data.squeeze(0)
        label = batch['label'].cuda()
        
        if in_dropout:
            data = F.dropout(data, p = in_dropout)
            
        event_time = batch['survival_time'].cuda()
        censorship = batch['censorship'].cuda()
        attn_mask = None
        

        logits = model(data)
            
        #(logits, y_prob, y_hat, a_raw, results_dict) = model(data)
        
        #print(logits.shape)
        #print(label.shape)
        #print(censorship.shape)
        out, log_dict = process_surv(logits, label, censorship, loss_fn)
        
        if out['loss'] is None:
            continue
        
        # backprop
        
        loss = out['loss']
        loss = loss / accum_steps
        loss.backward()
        
        if (batch_id + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        # survival specific metrics
        all_risk_scores.append(out['risk'].detach().cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())
        
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
    
            meters[key].update(val = val, n = len(data))
        
        if ((batch_id + 1) % print_every == 0) or (batch_id == len(loader) - 1):
            msg = [f'avg_{k}: {meter.avg:.4f}' for k, meter in meters.items()]
            msg = f'batch {batch_id} \t' + "\t".join(msg)

            print(msg)
            
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    
    c_index = concordance_index_censored(
        (1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol = 1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']

    msg = [f'{k}: {v:.3f}' for k, v in results.items()]
    print('\t'.join(msg))

    del all_risk_scores, all_censorships, all_event_times 
    gc.collect()
    torch.cuda.empty_cache()

    return results

def validate_step(model, loader, loss_fn, print_every = 100, dump_results = False, args = None):
    model.eval()
    
    meters = {}
    
    all_risk_scores, all_censorships, all_event_times = [], [], []
    
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            data = batch['img'].cuda()
            if len(data.shape) > 3:
                data = data.squeeze(0)
            label = batch['label'].cuda()
                
            event_time = batch['survival_time'].cuda()
            censorship = batch['censorship'].cuda()
            attn_mask = None
            

            logits = model(data)
            
            out, log_dict = process_surv(logits, label, censorship, loss_fn)
            
            # survival specific metrics
            all_risk_scores.append(out['risk'].detach().cpu().numpy())
            all_censorships.append(censorship.cpu().numpy())
            all_event_times.append(event_time.cpu().numpy())
            
            for key, val in log_dict.items():
                if key not in meters:
                    meters[key] = AverageMeter()
                
                meters[key].update(val = val, n = len(data))
            
            if ((batch_id + 1) % print_every == 0) or (batch_id == len(loader) - 1):
                msg = [f'avg_{k}: {meter.avg:.4f}' for k, meter in meters.items()]
                msg = f'batch {batch_id} \t' + "\t".join(msg)

                print(msg)
            
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    
    c_index = concordance_index_censored(
        (1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol = 1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})

    
    if args.recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        surv_loss_dict = loss_fn(logits = torch.tensor(all_risk_scores).unsqueeze(1),
                                times = torch.tensor(all_event_times).unsqueeze(1),
                                censorships = torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    msg = [f'{k}: {v:.3f}' for k, v in results.items()]
    print('\t'.join(msg))
    
    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['pids'] = loader.dataset.get_pids()

    del all_risk_scores, all_censorships, all_event_times 
    gc.collect()
    torch.cuda.empty_cache()
    
    
    return results, dumps

def train_cycle(train_loader, val_loader, args, fold = None):
    fold_metrics = {}
    if args.model_type == 'simple':
        model = LinearProbingSimple(feature_dim = 512, num_classes = args.n_label_bins)
    else:
        model = LinearProbingModel(feature_dim = 512, num_classes=args.n_label_bins)
    
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    writer = SummaryWriter(args.writer_dir, flush_secs = 15)
    
    lr_scheduler = get_lr_scheduler(args, optimizer, train_loader)
    
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha = args.nll_alpha)
        
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
        
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()
        
        
    for epoch in range(args.max_epochs):
        fold_metrics[epoch] = {}
        
        # train loop
        print('#' * 10, f'TRAIN Epoch: {epoch}/{args.max_epochs}, Fold: {fold}', '#' * 10)
        train_results = train_step(model, train_loader, optimizer, lr_scheduler, loss_fn, in_dropout = args.in_dropout,
                                                 print_every = args.print_every, accum_steps = args.accum_steps, args = args)
        
        writer = log_dict_tensorboard(writer, train_results, 'train/', epoch)
        
        # val loop
        print('#' * 11, f'VAL Epoch: {epoch}, Fold: {fold}', '#' * 11)
        
        val_results, _ = validate_step(
                        model, val_loader, loss_fn, print_every = args.print_every, args = args)
        
        
        writer = log_dict_tensorboard(writer, val_results, 'val/', epoch)
        
        save_checkpoint(
            model, 
            optimizer, epoch, 
            val_results['loss'], 
            train_results['loss'], 
            train_results['c_index'], 
            val_results['c_index'], 
            args
            )
        fold_metrics[epoch]['train'] = train_results
        fold_metrics[epoch]['val'] = val_results

    return fold_metrics


def run_test(test_loader, args):
    if args.model_type == 'simple':
        model = LinearProbingSimple(feature_dim = 512, n_classes = args.n_label_bins)
    else:
        model = LinearProbingModel(feature_dim = 512, n_classes=args.n_label_bins)
    
    state_dict = torch.load(args.model_checkpoint)
    model.load_state_dict(state_dict['model_state_dict'])
    model = model.cuda()

    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha = args.nll_alpha)
        
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
        
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()
    
    test_results, test_dumps = validate_step(
                        model, test_loader, loss_fn, print_every = args.print_every, args = args)
        
    return test_results, test_dumps

