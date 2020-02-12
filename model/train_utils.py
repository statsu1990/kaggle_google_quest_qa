import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
from torchcontrib.optim import SWA

from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd
import numpy as np


import warnings
warnings.simplefilter('ignore')

def save_log(loglist, filename):
    df = pd.DataFrame(loglist)
    df.columns = ['epoch', 'train_loss', 'train_score', 'test_loss', 'test_score']
    df.to_csv(filename)

def compute_spearmanr(original, preds):
    #score = 0
    #for i in range(30):
    #    score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)
    
    scores = []
    for i in range(30):
        scores.append(spearmanr(original[:, i], preds[:, i]).correlation)
    print(scores)
    return np.nanmean(scores)

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def pairwise_bce_logit_loss(outputs, targets):
    """
    outputs: logits
    """
    batch_size = outputs.size()[0]

    if batch_size < 3:
        pair_idx = np.arange(batch_size, dtype=np.int64)[::-1].copy()
        pair_idx = torch.from_numpy(pair_idx).cuda()
    else:
        pair_idx = torch.randperm(batch_size).cuda()

    diff_outputs = outputs - outputs[pair_idx]
    diff_targets = targets - targets[pair_idx]

    diff_targets = (1 + diff_targets) / 2

    loss = nn.BCEWithLogitsLoss()(diff_outputs, diff_targets)

    return loss

def pairwise_l1_logit_loss(outputs, targets):
    """
    outputs: logits
    """
    batch_size = outputs.size()[0]

    if batch_size < 3:
        pair_idx = np.arange(batch_size, dtype=np.int64)[::-1].copy()
        pair_idx = torch.from_numpy(pair_idx).cuda()
    else:
        pair_idx = torch.randperm(batch_size).cuda()

    diff_outputs = torch.sigmoid(outputs) - torch.sigmoid(outputs[pair_idx])
    diff_targets = targets - targets[pair_idx]

    loss = nn.L1Loss()(diff_outputs, diff_targets)

    return loss

def pairwise_l1_loss(outputs, targets):
    """
    """
    batch_size = outputs.size()[0]

    if batch_size < 3:
        pair_idx = np.arange(batch_size, dtype=np.int64)[::-1].copy()
        pair_idx = torch.from_numpy(pair_idx).cuda()
    else:
        pair_idx = torch.randperm(batch_size).cuda()

    #diff_outputs = torch.sigmoid(outputs) - torch.sigmoid(outputs[pair_idx])
    diff_outputs = outputs - outputs[pair_idx]
    diff_targets = targets - targets[pair_idx]

    loss = nn.L1Loss()(diff_outputs, diff_targets)

    return loss

def mseloss(outputs, targets):
    return torch.mean(torch.pow(torch.sub(outputs, targets), 2))

def wrapper_comb_point_pair_loss(pointwise_lossfunc, pairwise_lossfunc, pair_weight=1.0):
    def comb_point_pair_loss(outputs, targets):
        point_loss = pointwise_lossfunc(outputs, targets)
        pair_loss = pairwise_lossfunc(outputs, targets)
        loss = (1 - pair_weight) * point_loss + pair_weight * pair_loss
        return loss
    return comb_point_pair_loss

def train_model_v0(net, trainloader, validloader, epochs, lr, warmup_epoch=1, milestones=[5, 10], gamma=0.2):
    net = net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0

        for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
            optimizer.zero_grad()
            
            outputs = net(ids, masks, segments)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print(loss.item())

        print('Train Loss: %.3f' % (train_loss/(batch_idx+1),))
        return epoch, train_loss/(batch_idx+1)

    def test(epoch):
        net.eval()
        test_loss = 0

        with torch.no_grad():
            for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(validloader)):
                ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
                outputs = net(ids, masks, segments)
                loss = criterion(outputs, targets)

                test_loss += loss.item()

        print('Vali Loss: %.3f, ' % (test_loss/(batch_idx+1), ))
        return epoch, test_loss/(batch_idx+1)

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls = train(epoch)
        ep, ts_ls = test(epoch)
        loglist.append([ep, tr_ls, ts_ls])

    save_log(loglist, 'training_log.csv')

    return net

def train_model_v1(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2):
    net = net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
            outputs = net(ids, masks, segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(validloader)):
                ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
                outputs = net(ids, masks, segments)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net

def train_model_v2(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2):
    net = net.cuda()

    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
            outputs = net(ids, masks, segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(validloader)):
                ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
                outputs = net(ids, masks, segments)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net

def train_model_v3(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2):
    net = net.cuda()

    criterion = nn.L1Loss()
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
            outputs = net(ids, masks, segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(validloader)):
                ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
                outputs = net(ids, masks, segments)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net

def train_model_v4(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2):
    net = net.cuda()

    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
            outputs, _ = net(ids, masks, segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(validloader)):
                ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
                outputs, _ = net(ids, masks, segments)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net

def train_model_v5(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2):
    net = net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
            outputs, _ = net(ids, masks, segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (ids, masks, segments, targets) in enumerate(tqdm(validloader)):
                ids, masks, segments, targets = ids.cuda(), masks.cuda(), segments.cuda(), targets.cuda()
                outputs, _ = net(ids, masks, segments)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net


def train_model_sepQA_v1(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2, l2=0.0):
    net = net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.L1Loss()
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
            a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
            outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(validloader)):
                q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
                a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
                outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
                
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net

def train_model_sepQA_v1_1(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2, l2=0.0):
    net = net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    #criterion = mseloss #nn.MSELoss()
    #criterion = nn.L1Loss()
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
            a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
            outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(validloader)):
                q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
                a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
                outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
                
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net

def train_model_sepQA_v1_2_mix(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2, l2=0.0):
    """
    mixup
    """
    net = net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    #criterion = mseloss #nn.MSELoss()
    #criterion = nn.L1Loss()
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
            a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
            outputs, _, mix_idx, mix_rate = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
            
            targets = mix_rate * targets + (1 - mix_rate) * targets[mix_idx]

            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(validloader)):
                q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
                a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
                outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
                
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net

def train_model_sepQA_v1_3(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2, l2=0.0, tg_indexs=None):
    net = net.cuda()

    #criterion = nn.BCEWithLogitsLoss()
    criterion = MultiLossWrapper(nn.BCEWithLogitsLoss(), tg_indexs)
    #criterion = mseloss #nn.MSELoss()
    #criterion = nn.L1Loss()
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
            a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
            outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(validloader)):
                q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
                a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
                outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
                
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net

def train_model_sepQA_v1_4(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2, l2=0.0, tg_indexs=None):
    net = net.cuda()

    #criterion = nn.BCEWithLogitsLoss()
    criterion = MultiLossWrapper_AllAverage(nn.BCEWithLogitsLoss(), tg_indexs)
    #criterion = mseloss #nn.MSELoss()
    #criterion = nn.L1Loss()
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
            a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
            outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(validloader)):
                q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
                a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
                outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
                
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net


def MultiLossWrapper_AllAverage(loss_func, tg_indexs):
    def LossFunc(outputs, targets):
        num = outputs.size()[1]
        loss = 0

        ave_output = None

        for i in range(num):
            if i in tg_indexs:
                if ave_output is None:
                    ave_output = outputs[:,i]
                else:
                    ave_output += outputs[:,i]
        ave_output = ave_output / len(tg_indexs)

        for i in range(num):
            if i in tg_indexs:
                loss += loss_func(ave_output, targets[:,i])
        loss = loss / len(tg_indexs)
        return loss
    return LossFunc


# pair
def train_model_sepQA_v2(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2, pair_w=None, l2=0.0):
    """
    pair
    """
    net = net.cuda()

    if pair_w is None:
        PAIR_WEIHGT = 1.0
    else:
        PAIR_WEIHGT = pair_w
    criterion = wrapper_comb_point_pair_loss(nn.BCEWithLogitsLoss(), pairwise_l1_logit_loss, PAIR_WEIHGT)
    #criterion = wrapper_comb_point_pair_loss(nn.L1Loss(), pairwise_l1_loss, PAIR_WEIHGT)
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.L1Loss()
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0,)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
            a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
            outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(validloader)):
                q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
                a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
                outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
                
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net

# swa
def train_model_sepQA_v3_1(net, trainloader, validloader, epochs, lr, 
                           swa_start_epoch, swa_freq_step,
                           grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2, l2=0.0, 
                           ):
    net = net.cuda()

    criterion = nn.BCEWithLogitsLoss()
    base_optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
    optimizer = SWA(base_optimizer)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
            a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
            outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                if epoch >= swa_start_epoch and ((batch_idx + 1) % (grad_accum_steps * swa_freq_step)) == 0:
                    optimizer.update_swa()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(validloader)):
                q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
                a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
                outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
                
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    optimizer.swap_swa_sgd()
    ep, ts_ls, ts_sc = test(epochs)
    loglist.append([ep, -1, -1, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net

# classification
def MultiLossWrapper(loss_func, tg_indexs):
    def LossFunc(outputs, targets):
        num = outputs.size()[1]
        loss = 0

        if tg_indexs is None:
            for i in range(num):
                loss += loss_func(outputs[:,i], targets[:,i])
            loss = loss / num
        else:
            for i in range(num):
                if i in tg_indexs:
                    loss += loss_func(outputs[:,i], targets[:,i])
            loss = loss / len(tg_indexs)
        return loss
    return LossFunc

def train_model_sepQA_v4_1(net, trainloader, validloader, epochs, lr, grad_accum_steps=1, warmup_epoch=1, milestones=[5, 10], gamma=0.2, l2=0.0, tg_indexs=None):
    net = net.cuda()

    criterion = MultiLossWrapper(nn.CrossEntropyLoss(), tg_indexs)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) #learning rate decay
    warmup_scheduler = WarmUpLR(optimizer, len(trainloader) * warmup_epoch)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_score = 0

        preds = []
        original = []
        optimizer.zero_grad()
        for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(trainloader)):
            if epoch < warmup_epoch:
                warmup_scheduler.step()

            q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
            a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
            outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
            
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            #print(loss.item() * grad_accum_steps)
            with torch.no_grad():
                preds.append(outputs.max(2)[1].cpu().numpy()) # label
                original.append(targets.cpu().numpy())

        train_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))

        print('Train Loss: %.3f, Score: %.3f' % (train_loss/(batch_idx+1), train_score))
        return epoch, train_loss/(batch_idx+1), train_score

    def test(epoch):
        net.eval()
        test_loss = 0
        test_score = 0

        preds = []
        original = []
        with torch.no_grad():
            for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(validloader)):
                q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
                a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
                outputs, _ = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)
                
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds.append(outputs.max(2)[1].cpu().numpy())
                original.append(targets.cpu().numpy())

        test_score = compute_spearmanr(np.concatenate(original), np.concatenate(preds))
        print('Vali Loss: %.3f, Score: %.3f' % (test_loss/(batch_idx+1), test_score))
        return epoch, test_loss/(batch_idx+1), test_score

    loglist = []
    for epoch in range(0, epochs):
        if epoch > warmup_epoch - 1:
            scheduler.step(epoch)

        ep, tr_ls, tr_sc = train(epoch)
        ep, ts_ls, ts_sc = test(epoch)
        loglist.append([ep, tr_ls, tr_sc, ts_ls, ts_sc])

    save_log(loglist, 'training_log.csv')

    return net
