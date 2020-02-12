import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from scipy.stats import spearmanr
from scipy import stats
import pandas as pd
import numpy as np


import os
import pickle

import warnings
warnings.simplefilter('ignore')

def compute_spearmanr(original, preds):
    #score = 0
    #for i in range(30):
    #    score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)
    
    scores = []
    for i in range(30):
        scores.append(spearmanr(original[:, i], preds[:, i]).correlation)

    print(scores)

    return np.nanmean(scores)

def compute_modi_preds(preds, weight):
    modi_regs_path = '../input/modi_score_regs/'

    modi_preds = np.zeros_like(preds)

    for i in range(30):
        tg_idx = i
        x_idx = np.delete(np.arange(30), obj=i, axis=0)
        
        reg_file = modi_regs_path + 'reg_using_target_' + str(tg_idx) + '.pickle'
        with open(reg_file, 'rb') as f:
            reg = pickle.load(f)

        modi_preds[:,tg_idx] = reg.predict(preds[:,x_idx])
        
    modi_preds = weight * preds + (1 - weight) * modi_preds

    return modi_preds

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def pred_score(net, dataloader):
    net = net.cuda()
    
    net.eval()
    test_score = 0

    preds = []
    original = []
    with torch.no_grad():
        for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(dataloader)):
            ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
            outputs = net(ids, masks, segments)

            preds.append(outputs.cpu().numpy())
            original.append(targets.cpu().numpy())

    score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
    print('Score: %.3f' % (score,))
    return score

def pred_score2(net, dataloader):
    net = net.cuda()
    
    net.eval()
    test_score = 0

    preds = []
    original = []
    with torch.no_grad():
        for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(dataloader)):
            ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
            outputs = net(ids, masks, segments)
            outputs = outputs[0]

            preds.append(outputs.cpu().numpy())
            original.append(targets.cpu().numpy())

    score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
    print('Score: %.5f' % (score,))

    a1 = np.concatenate(original)
    b1 = np.concatenate(preds)
    c1 = np.concatenate((a1, b1), axis=1)
    pd.DataFrame(c1).to_csv('orig_pred.csv')

    return score

def pred_score_sepQA_1(net, dataloader):
    net = net.cuda()
    
    net.eval()
    test_score = 0

    preds = []
    preds_hidden = []
    original = []
    with torch.no_grad():
        for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(dataloader)):
            q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
            a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
            outputs, hidden_outpus = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)

            preds.append(outputs.cpu().numpy())
            preds_hidden.append(hidden_outpus.cpu().numpy())
            original.append(targets.cpu().numpy())

    original = np.concatenate(original)
    preds = np.concatenate(preds)
    preds = sigmoid(preds)
    preds_hidden = np.concatenate(preds_hidden)
    original_preds = np.concatenate((original, preds), axis=1)
    
    score = compute_spearmanr(original, preds)
    print('Score: %.5f' % (score,))

    #pred_rank = np.apply_along_axis(stats.mstats.rankdata, axis=0, arr=preds) / len(preds)
    #score = compute_spearmanr(original, pred_rank)
    #print('Score: %.5f' % (score,))

    pd.DataFrame(original).to_csv('original.csv')
    pd.DataFrame(preds).to_csv('preds.csv')
    pd.DataFrame(preds_hidden).to_csv('preds_hidden.csv')
    pd.DataFrame(original_preds).to_csv('original_preds.csv')

    return score
