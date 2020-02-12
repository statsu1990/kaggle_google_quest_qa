from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

import os
from sklearn.model_selection import ShuffleSplit
import numpy as np

import config
import data_utils as du
from model import train_utils as tu
from model import torch_data_utils as tdu
from model import pred_utils as pu
from model.ml_stratifiers import MultilabelStratifiedKFold

class BertUtils:
    @staticmethod
    def save_bert_tokenizer(pretrained_model, save_path):
        tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        tokenizer.save_pretrained(save_path)
        return

    @staticmethod
    def get_bert_tokenizer(save_path):
        tokenizer = BertTokenizer.from_pretrained(save_path)
        return tokenizer

    @staticmethod
    def save_bert_model(pretrained_model, save_path):
        model = BertModel.from_pretrained(pretrained_model)
        model.save_pretrained(save_path)
        return

    @staticmethod
    def get_bert_model(save_path):
        model = BertModel.from_pretrained(save_path)
        return model

class Bert_v0(nn.Module):
    def __init__(self, pretrained_model, num_bert_last_hidden=768, num_target=30):
        super(Bert_v0, self).__init__()

        self.num_bert_last_hidden = num_bert_last_hidden
        self.num_target = num_target

        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.num_bert_last_hidden, self.num_target)

    def forward(self, id, mask, seg):
        oup = self.pretrained_model(id, mask, seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        hidden_oup = torch.mean(oup, dim=1)

        oup = self.dropout(hidden_oup)
        oup = self.linear(oup)

        return oup, hidden_oup

class Bert_v1_0(nn.Module):
    def __init__(self, pretrained_model, num_bert_last_hidden=768, num_q_target=21, num_a_target=9, drop_p=0.1):
        super(Bert_v1_0, self).__init__()

        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(p=drop_p)
        self.q_linear = nn.Linear(num_bert_last_hidden, num_q_target)
        self.a_linear = nn.Linear(num_bert_last_hidden*2, num_a_target)

    def forward(self, q_id, q_mask, q_seg, a_id, a_mask, a_seg):
        oup = self.pretrained_model(q_id, q_mask, q_seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        q_hidden_oup = torch.mean(oup, dim=1)
        q_oup = self.dropout(q_hidden_oup)

        oup = self.pretrained_model(a_id, a_mask, a_seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        a_hidden_oup = torch.mean(oup, dim=1)
        a_oup = self.dropout(a_hidden_oup)

        # q:21, a:9
        q_result = self.q_linear(q_oup)
        a_result = self.a_linear(torch.cat([q_oup, a_oup], dim=1))

        return torch.cat([q_result, a_result], dim=1), torch.cat([q_hidden_oup, a_hidden_oup], dim=1)

    def freeze_pretrained_model(self):
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        return

class Bert_v1_0_mix(nn.Module):
    def __init__(self, pretrained_model, num_bert_last_hidden=768, num_q_target=21, num_a_target=9, drop_p=0.1, mixup_alpha=None):
        super(Bert_v1_0_mix, self).__init__()

        self.mixup_alpha = mixup_alpha

        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(p=drop_p)
        self.q_linear = nn.Linear(num_bert_last_hidden, num_q_target)
        self.a_linear = nn.Linear(num_bert_last_hidden*2, num_a_target)

    def forward(self, q_id, q_mask, q_seg, a_id, a_mask, a_seg):
        oup = self.pretrained_model(q_id, q_mask, q_seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        q_hidden_oup = torch.mean(oup, dim=1)

        oup = self.pretrained_model(a_id, a_mask, a_seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        a_hidden_oup = torch.mean(oup, dim=1)

        # mixup
        if self.training:
            if self.mixup_alpha is not None:
                q_hidden_oup, a_hidden_oup, mix_idx, mix_rate = self.mixup_data(q_hidden_oup, a_hidden_oup, self.mixup_alpha)

        q_oup = self.dropout(q_hidden_oup)
        a_oup = self.dropout(a_hidden_oup)


        # q:21, a:9
        q_result = self.q_linear(q_oup)
        a_result = self.a_linear(torch.cat([q_oup, a_oup], dim=1))

        if self.training:
            if self.mixup_alpha is not None:
                return torch.cat([q_result, a_result], dim=1), torch.cat([q_hidden_oup, a_hidden_oup], dim=1), mix_idx, mix_rate
        else:
            return torch.cat([q_result, a_result], dim=1), torch.cat([q_hidden_oup, a_hidden_oup], dim=1)

    def mixup_data(self, x1, x2, alpha):
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        batch_size = x1.size()[0]
        index = torch.randperm(batch_size).cuda()

        mixed_x1 = lam * x1 + (1 - lam) * x1[index,:]
        mixed_x2 = lam * x2 + (1 - lam) * x2[index,:]
        
        return mixed_x1, mixed_x2, index, lam

    def freeze_pretrained_model(self):
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        return

class Bert_v1_0_classify(nn.Module):
    def __init__(self, pretrained_model, num_class=5, num_bert_last_hidden=768, num_q_target=21, num_a_target=9, drop_p=0.1):
        super(Bert_v1_0_classify, self).__init__()

        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(p=drop_p)

        self.q_linears = nn.ModuleList(nn.Linear(num_bert_last_hidden, num_class) for i in range(num_q_target))
        self.a_linears = nn.ModuleList(nn.Linear(num_bert_last_hidden*2, num_class) for i in range(num_a_target))

    def forward(self, q_id, q_mask, q_seg, a_id, a_mask, a_seg):
        # q out
        oup = self.pretrained_model(q_id, q_mask, q_seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        q_hidden_oup = torch.mean(oup, dim=1)
        q_oup = self.dropout(q_hidden_oup)

        # a out
        oup = self.pretrained_model(a_id, a_mask, a_seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        a_hidden_oup = torch.mean(oup, dim=1)
        a_oup = self.dropout(a_hidden_oup)

        # results [(batch, class), (batch, class), ...]
        results = []
        # q:21
        for q_linear in self.q_linears:
            results.append(torch.unsqueeze(q_linear(q_oup), 1))
        # a:9
        for a_linear in self.a_linears:
            results.append(torch.unsqueeze(a_linear(torch.cat([q_oup, a_oup], dim=1)), 1))
        
        return torch.cat(results, dim=1), torch.cat([q_hidden_oup, a_hidden_oup], dim=1)

    def freeze_pretrained_model(self):
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        return

class Bert_v1_1(nn.Module):
    def __init__(self, pretrained_model, num_bert_last_hidden=768):
        super(Bert_v1_1, self).__init__()

        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(num_bert_last_hidden*2, 30)

    def forward(self, q_id, q_mask, q_seg, a_id, a_mask, a_seg):
        oup = self.pretrained_model(q_id, q_mask, q_seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        q_hidden_oup = torch.mean(oup, dim=1)

        oup = self.pretrained_model(a_id, a_mask, a_seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        a_hidden_oup = torch.mean(oup, dim=1)
        
        oup = self.linear(self.dropout(torch.cat([q_hidden_oup, a_hidden_oup], dim=1)))
        
        return oup, torch.cat([q_hidden_oup, a_hidden_oup], dim=1)

class Bert_v1_2(nn.Module):
    def __init__(self, pretrained_model, num_bert_last_hidden=768, num_q_target=21, num_a_target=9, drop_p=0.1):
        super(Bert_v1_2, self).__init__()

        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(p=drop_p)
        self.q_linear = nn.Linear(num_bert_last_hidden, num_q_target)
        self.a_linear = nn.Linear(num_bert_last_hidden, num_a_target)

    def forward(self, q_id, q_mask, q_seg, a_id, a_mask, a_seg):
        oup = self.pretrained_model(q_id, q_mask, q_seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        q_hidden_oup = torch.mean(oup, dim=1)
        q_oup = self.dropout(q_hidden_oup)

        oup = self.pretrained_model(a_id, a_mask, a_seg) # (batch, seq length, self.num_bert_last_hidden)
        oup = oup[0]
        a_hidden_oup = torch.mean(oup, dim=1)
        a_oup = self.dropout(a_hidden_oup)

        # q:21, a:9
        q_result = self.q_linear(q_oup)
        a_result = self.a_linear(a_oup)

        return torch.cat([q_result, a_result], dim=1), torch.cat([q_hidden_oup, a_hidden_oup], dim=1)

def test1():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    #model = BertUtils.get_bert_model(pretrained_model_path) # output((batch, seq length, 1024), (batch, 1024))
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v2.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v2.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v2.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 3
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v0(BertUtils.get_bert_model(pretrained_model_path), num_bert_last_hidden=768, num_target=30)

            # train
            epochs = 10
            lr = 2e-5 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 10
            model = tu.train_model_v4(model, train_loader, val_loader, epochs, lr, grad_accum_steps, warmup_epoch=1, milestones=[5, 10], gamma=0.5)
            torch.save(model.state_dict(), 'bert_model')

    return

def test2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    #model = BertUtils.get_bert_model(pretrained_model_path) # output((batch, seq length, 1024), (batch, 1024))
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v2.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v2.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v2.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 3
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v0(BertUtils.get_bert_model(pretrained_model_path), num_bert_last_hidden=768, num_target=30)

            # train
            epochs = 10
            lr = 2e-5 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 10
            model = tu.train_model_v5(model, train_loader, val_loader, epochs, lr, grad_accum_steps, warmup_epoch=1, milestones=[5, 10], gamma=0.5)
            torch.save(model.state_dict(), 'bert_model')

    return

def test3():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    #model = BertUtils.get_bert_model(pretrained_model_path) # output((batch, seq length, 1024), (batch, 1024))
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v2.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v2.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v2.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 3
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v0(BertUtils.get_bert_model(pretrained_model_path), num_bert_last_hidden=768, num_target=30)

            # train
            epochs = 10
            lr = 1e-4 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 10
            model = tu.train_model_v5(model, train_loader, val_loader, epochs, lr, grad_accum_steps, warmup_epoch=1, milestones=[10, 20], gamma=0.5)
            torch.save(model.state_dict(), 'bert_model')

    return


def test_sepQA_1():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 5e-5 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 16

            model = tu.train_model_sepQA_v1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5)
            torch.save(model.state_dict(), 'bert_model_1')

    return

def test_sepQA_2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 1e-4 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5)
            torch.save(model.state_dict(), 'bert_model_2')

    return

def test_sepQA_2_5fold():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if True: #ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 1e-4 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5)
            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return

def test_sepQA_3():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=False, clip_output=[0.05, 0.95])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 1e-4 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001)
            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return

# sepQA_3 : relative rank
def test_sepQA_3_1():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if True:#ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001)
            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return

def test_sepQA_3_1_2():
    """
    test_sepQA_3_1_2 : relative rank, clip_output, drop_p=0.2
    """
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512
    DROP_P = 0.2

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.05, 0.95])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path), drop_p=DROP_P)

            # train
            epochs = 5
            lr = 1e-4 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001)
            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return

def test_sepQA_3_1_3():
    """
    test_sepQA_3_1_3 : relative rank, clip_output, drop_p=0.2, mixup_alpha=0.2
    """
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512
    DROP_P = 0.1
    MIXUP_ALPHA = 0.2

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0_mix(BertUtils.get_bert_model(pretrained_model_path), drop_p=DROP_P, mixup_alpha=MIXUP_ALPHA)

            # train
            epochs = 5
            lr = 1e-4 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_2_mix(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001)
            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return

# MultilabelStratifiedKFold
def test_sepQA_4_1():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld > 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001)
            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return

def test_sepQA_4_1_2_comb1():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[0,1,2,4,5,6,7,8,10,11,16,18,20,22,26,28])
            torch.save(model.state_dict(), 'Bert_v1_0_comb1_model_fold' + str(ifld))

    return

def test_sepQA_4_1_2_comb1_v2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[0,1,2,3,4,5,6,7,8,10,11,16,17,18,26,27,28])
            torch.save(model.state_dict(), 'Bert_v1_0_comb1_model_fold' + str(ifld))

    return

def test_sepQA_4_1_2_comb2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[3,17,21,23,24,25,27,29])
            torch.save(model.state_dict(), 'Bert_v1_0_comb2_model_fold' + str(ifld))

    return

def test_sepQA_4_1_2_comb2_v2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[20,21,22,23,24,25,29])
            torch.save(model.state_dict(), 'Bert_v1_0_comb2_model_fold' + str(ifld))

    return

def test_sepQA_4_1_2_comb3():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[9])
            torch.save(model.state_dict(), 'Bert_v1_0_comb3_model_fold' + str(ifld))

    return

def test_sepQA_4_1_2_comb4():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 5e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[29])
            torch.save(model.state_dict(), 'Bert_v1_0_comb4_model_fold' + str(ifld))

    return

def test_sepQA_4_1_2_comb5():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=False, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 5e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[21,23,24])
            torch.save(model.state_dict(), 'Bert_v1_0_comb5_model_fold' + str(ifld))

    return

def test_sepQA_4_1_2_comb5_v2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_2(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 10
            lr = 5e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[20], gamma=0.5, l2=0.0001, tg_indexs=[21,23,24])
            torch.save(model.state_dict(), 'Bert_v1_0_comb5_v2_model_fold' + str(ifld))

    return

def test_sepQA_4_1_2_comb5_v3():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    #fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
    #for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 5e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_4(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[21,23,24])
            torch.save(model.state_dict(), 'Bert_v1_0_comb5_v3_model_fold' + str(ifld))

    return


def test_sepQA_4_1_2_comb6():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_2(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[20], gamma=0.5, l2=0.0001, tg_indexs=None)
            torch.save(model.state_dict(), 'Bert_v1_0_comb6_model_fold' + str(ifld))

    return

# random sampler
def test_sepQA_4_2_1_comb4():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            # sampler weight
            lb = train_df.values[tr_idxs,11:]
            cond = []
            cond.append((lb[:,21] < np.max(lb[:,21]))[:,np.newaxis])
            cond.append((lb[:,23] < np.max(lb[:,23]))[:,np.newaxis])
            cond.append((lb[:,24] < np.max(lb[:,24]))[:,np.newaxis])
            cond = np.concatenate(cond, axis=1)
            cond = np.any(cond, axis=1)
            weights = np.ones(len(tr_idxs))
            weights[cond] = 1 * 1.5
            weights = weights / np.sum(weights)

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size, weights=weights)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 3e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[21,23,24])
            torch.save(model.state_dict(), 'Bert_v1_0_comb4_1_model_fold' + str(ifld))

    return

def test_sepQA_4_2_1_comb4_2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            # sampler weight
            lb = train_df.values[tr_idxs,11:]
            cond = []
            cond.append((lb[:,21] < np.max(lb[:,21]))[:,np.newaxis])
            cond.append((lb[:,23] < np.max(lb[:,23]))[:,np.newaxis])
            cond.append((lb[:,24] < np.max(lb[:,24]))[:,np.newaxis])
            cond = np.concatenate(cond, axis=1)
            cond = np.any(cond, axis=1)
            weights = np.ones(len(tr_idxs))
            weights[cond] = 1 * 2
            weights = weights / np.sum(weights)

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size, weights=weights)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 3e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[21,23,24])
            torch.save(model.state_dict(), 'Bert_v1_0_comb4_1_model_fold' + str(ifld))

    return

def test_sepQA_4_2_1_comb5_1():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            # sampler weight
            lb = train_df.values[tr_idxs,11:]
            cond = []
            cond.append((lb[:,21] < np.max(lb[:,21]))[:,np.newaxis])
            #cond.append((lb[:,23] < np.max(lb[:,23]))[:,np.newaxis])
            #cond.append((lb[:,24] < np.max(lb[:,24]))[:,np.newaxis])
            cond = np.concatenate(cond, axis=1)
            cond = np.any(cond, axis=1)
            weights = np.ones(len(tr_idxs))
            weights[cond] = 1 * 4
            weights = weights / np.sum(weights)

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size, weights=weights)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 3e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[21])
            torch.save(model.state_dict(), 'Bert_v1_0_comb5_1_model_fold' + str(ifld))

    return

def test_sepQA_4_2_1_comb5_2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            # sampler weight
            lb = train_df.values[tr_idxs,11:]
            cond = []
            #cond.append((lb[:,21] < np.max(lb[:,21]))[:,np.newaxis])
            cond.append((lb[:,23] < np.max(lb[:,23]))[:,np.newaxis])
            #cond.append((lb[:,24] < np.max(lb[:,24]))[:,np.newaxis])
            cond = np.concatenate(cond, axis=1)
            cond = np.any(cond, axis=1)
            weights = np.ones(len(tr_idxs))
            weights[cond] = 1 * 4
            weights = weights / np.sum(weights)

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size, weights=weights)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 3e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[23])
            torch.save(model.state_dict(), 'Bert_v1_0_comb5_2_model_fold' + str(ifld))

    return

def test_sepQA_4_2_1_comb5_3():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            # sampler weight
            lb = train_df.values[tr_idxs,11:]
            cond = []
            #cond.append((lb[:,21] < np.max(lb[:,21]))[:,np.newaxis])
            #cond.append((lb[:,23] < np.max(lb[:,23]))[:,np.newaxis])
            cond.append((lb[:,24] < np.max(lb[:,24]))[:,np.newaxis])
            cond = np.concatenate(cond, axis=1)
            cond = np.any(cond, axis=1)
            weights = np.ones(len(tr_idxs))
            weights[cond] = 1 * 4
            weights = weights / np.sum(weights)

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size, weights=weights)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 3
            lr = 3e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v1_3(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, l2=0.0001, tg_indexs=[24])
            torch.save(model.state_dict(), 'Bert_v1_0_comb5_2_model_fold' + str(ifld))

    return

# SWA
def test_sepQA_4_2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=[0.01, 0.99])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16
            swa_star_epoch = 3
            swa_freq_step = 5

            model = tu.train_model_sepQA_v3_1(model, train_loader, val_loader, epochs, lr, 
                           swa_start_epoch=swa_star_epoch, swa_freq_step=swa_freq_step,
                           grad_accum_steps=grad_accum_steps, warmup_epoch=1, milestones=[3, 5], gamma=0.1, l2=0.0001, 
                           )

            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return

# classify
def test_sepQA_5_1():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=None, hard_target_num=5)
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0_classify(BertUtils.get_bert_model(pretrained_model_path), num_class=5)

            # train
            epochs = 20
            lr = 5e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v4_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[30], gamma=0.5, l2=0.0001)
            torch.save(model.state_dict(), 'Bert_v1_0_clasify_model_fold' + str(ifld))

    return

def test_sepQA_5_1_3():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=None, hard_target_num=4)
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0_classify(BertUtils.get_bert_model(pretrained_model_path), num_class=4)

            # train
            epochs = 3
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v4_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[30], gamma=0.5, l2=0.0001)
            torch.save(model.state_dict(), 'Bert_v1_0_clasify_model_fold' + str(ifld))

    return

def test_sepQA_5_1_3_1():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=None, hard_target_num=4)
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0_classify(BertUtils.get_bert_model(pretrained_model_path), num_class=4)

            # train
            epochs = 7
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v4_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[30], gamma=0.5, l2=0.0001, tg_indexs=[12, 13, 14, 15])
            torch.save(model.state_dict(), 'Bert_v1_0_clasify_model_fold' + str(ifld))

    return

def test_sepQA_5_1_3_2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=None, hard_target_num=4)
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0_classify(BertUtils.get_bert_model(pretrained_model_path), num_class=4)

            # train
            epochs = 7
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v4_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[30], gamma=0.5, l2=0.0001, tg_indexs=[13])
            torch.save(model.state_dict(), 'Bert_v1_0_clasify_model_fold' + str(ifld))

    return

def test_sepQA_5_1_4_comb1():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=None, hard_target_num=4)
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0_classify(BertUtils.get_bert_model(pretrained_model_path), num_class=4)

            # train
            epochs = 3
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v4_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[30], gamma=0.5, l2=0.0001, tg_indexs=[12,14,15])
            torch.save(model.state_dict(), 'Bert_v1_0_clasify_comb1_model_fold' + str(ifld))

    return

def test_sepQA_5_1_4_comb2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=None, hard_target_num=4)
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0_classify(BertUtils.get_bert_model(pretrained_model_path), num_class=4)

            # train
            epochs = 2
            lr = 1e-4 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v4_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[30], gamma=0.5, l2=0.0001, tg_indexs=[13])
            torch.save(model.state_dict(), 'Bert_v1_0_clasify_comb2_model_fold' + str(ifld))

    return

def test_sepQA_5_1_4_comb2_v2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=None, hard_target_num=5)
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0_classify(BertUtils.get_bert_model(pretrained_model_path), num_class=5)

            # train
            epochs = 4
            lr = 5e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v4_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[30], gamma=0.5, l2=0.0001, tg_indexs=[13])
            torch.save(model.state_dict(), 'Bert_v1_0_clasify_comb2_model_fold' + str(ifld))

    return

def test_sepQA_5_1_4_comb3():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=None, hard_target_num=5)
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0_classify(BertUtils.get_bert_model(pretrained_model_path), num_class=5)

            # train
            epochs = 5
            lr = 5e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v4_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[30], gamma=0.5, l2=0.0001, tg_indexs=[2])
            torch.save(model.state_dict(), 'Bert_v1_0_clasify_comb3_model_fold' + str(ifld))

    return

def test_sepQA_5_1_4_comb4():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(to_relative_rank=True, clip_output=None, hard_target_num=2)
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld < 3:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0_classify(BertUtils.get_bert_model(pretrained_model_path), num_class=2)

            # train
            epochs = 5
            lr = 5e-5 # 1e-4, 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v4_1(model, train_loader, val_loader, epochs, lr, grad_accum_steps, 
                                            warmup_epoch=1, milestones=[30], gamma=0.5, l2=0.0001, tg_indexs=[21,23,24])
            torch.save(model.state_dict(), 'Bert_v1_0_clasify_comb4_model_fold' + str(ifld))

    return

# pair
def test_sepQA_2_pair_1():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 10
            lr = 1e-4 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v2(model, train_loader, val_loader, epochs, lr, grad_accum_steps, warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5)
            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return

def test_sepQA_2_pair_1_2():
    """
    test_sepQA_2_pair_1_2: clip output
    """
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data(clip_output=[0.05, 0.95])
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

            # train
            epochs = 5
            lr = 1e-4 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v2(model, train_loader, val_loader, epochs, lr, grad_accum_steps, warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5, pair_w=1.0)
            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return

def test_sepQA_2_pair_2():
    model_path = os.path.join('result', '20020102_bert_v1_0_uncased_bce_lr0.0001_seed2020_bs16', 'bert_model_2')

    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 2
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))
            model.load_state_dict(torch.load(model_path))

            # train
            epochs = 10
            lr = 1e-4 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 8 #16

            model = tu.train_model_sepQA_v2(model, train_loader, val_loader, epochs, lr, grad_accum_steps, warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5)
            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return

def test_sepQA_2_pair_3():
    model_path = os.path.join('result', '20020102_bert_v1_0_uncased_bce_lr0.0001_seed2020_bs16', 'bert_model_2')

    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    #BertUtils.save_bert_tokenizer(pretrained_model, pretrained_tokenizer_path)
    #BertUtils.save_bert_model(pretrained_model, pretrained_model_path)

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 32
            val_batch_size = 16
            train_loader = tdu.get_dataloader(train_ds, train_batch_size)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #
            model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))
            model.load_state_dict(torch.load(model_path))
            model = Bert_v1_0(model.pretrained_model)
            model.freeze_pretrained_model()

            # train
            epochs = 10
            lr = 1e-3 # 5e-5, 3e-5, 2e-5
            grad_accum_steps = 1 #16

            model = tu.train_model_sepQA_v2(model, train_loader, val_loader, epochs, lr, grad_accum_steps, warmup_epoch=1, milestones=[3, 5, 7, 9], gamma=0.5)
            torch.save(model.state_dict(), 'Bert_v1_0_model_fold' + str(ifld))

    return


def check_score_test_sepQA_2():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    model_path = os.path.join('result', '20020102_bert_v1_0_uncased_bce_lr0.0001_seed2020_bs16', 'bert_model_2')
    model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))
    model.load_state_dict(torch.load(model_path))
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # training
    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)

    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 32
            val_batch_size = 32
            train_loader = tdu.get_dataloader(train_ds, train_batch_size, shuffle=False)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            #score = pu.pred_score_sepQA_1(model, train_loader)
            score = pu.pred_score_sepQA_1(model, val_loader)

    return

def check_score_test_sepQA_3():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    model_path = [
        os.path.join('result', '20020302_bert_v1_0_uncased_bce_relativerank_lr0.0001_seed2020_bs16_smooth001_epoch4_5fold', 'Bert_v1_0_model_fold0'),
        os.path.join('result', '20020302_bert_v1_0_uncased_bce_relativerank_lr0.0001_seed2020_bs16_smooth001_epoch4_5fold', 'Bert_v1_0_model_fold1'),
        os.path.join('result', '20020302_bert_v1_0_uncased_bce_relativerank_lr0.0001_seed2020_bs16_smooth001_epoch4_5fold', 'Bert_v1_0_model_fold2'),
        os.path.join('result', '20020302_bert_v1_0_uncased_bce_relativerank_lr0.0001_seed2020_bs16_smooth001_epoch4_5fold', 'Bert_v1_0_model_fold3'),
        os.path.join('result', '20020302_bert_v1_0_uncased_bce_relativerank_lr0.0001_seed2020_bs16_smooth001_epoch4_5fold', 'Bert_v1_0_model_fold4'),
        ]

    model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # test
    score_tr = 0
    score_vl = 0
    counter = 0

    fld = ShuffleSplit(n_splits=5, test_size=.2, random_state=2020)
    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df)):
        if ifld == 0:
            train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 32
            val_batch_size = 32
            train_loader = tdu.get_dataloader(train_ds, train_batch_size, shuffle=False)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            model.load_state_dict(torch.load(model_path[ifld]))

            print('fold {0}'.format(ifld))
            #print('train')
            #score_tr += pu.pred_score_sepQA_1(model, train_loader)
            print('val')
            score_vl += pu.pred_score_sepQA_1(model, val_loader)
            counter += 1

    print('average')
    print('train')
    score_tr = score_tr / counter
    print(score_tr)
    print('val')
    score_vl = score_vl / counter
    print(score_vl)

    return

def check_score_test_sepQA_4():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    model_path = [
        os.path.join('result', '20020701_bert_v1_0_msk', 'Bert_v1_0_model_fold0'),
        os.path.join('result', '20020701_bert_v1_0_msk', 'Bert_v1_0_model_fold1'),
        os.path.join('result', '20020701_bert_v1_0_msk', 'Bert_v1_0_model_fold2'),
        os.path.join('result', '20020701_bert_v1_0_msk', 'Bert_v1_0_model_fold3'),
        os.path.join('result', '20020701_bert_v1_0_msk', 'Bert_v1_0_model_fold4'),
        ]

    model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # test
    score_tr = 0
    score_vl = 0
    counter = 0

    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020)
    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if True: #ifld == 0:
            #train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 32
            val_batch_size = 32
            #train_loader = tdu.get_dataloader(train_ds, train_batch_size, shuffle=False)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            model.load_state_dict(torch.load(model_path[ifld]))

            print('fold {0}'.format(ifld))
            #print('train')
            #score_tr += pu.pred_score_sepQA_1(model, train_loader)
            print('val')
            score_vl += pu.pred_score_sepQA_1(model, val_loader)
            counter += 1

    print('average')
    print('train')
    score_tr = score_tr / counter
    print(score_tr)
    print('val')
    score_vl = score_vl / counter
    print(score_vl)

    return

def check_score_test_sepQA_5():
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    model_path = [
        os.path.join('result', '20021001_bert_v1_0_comb5', 'Bert_v1_0_comb5_model_fold0'),
        #os.path.join('result', '20021001_bert_v1_0_comb5', 'Bert_v1_0_comb5_model_fold1'),
        #os.path.join('result', '20021001_bert_v1_0_comb5', 'Bert_v1_0_comb5_model_fold2'),
        ]

    model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))
    
    # raw data
    train_df = du.InputData.get_train_data()
    test_df = du.InputData.get_test_data()

    # bert input
    train_bert_inp = du.BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
    train_bert_label = du.BertData_v3.compute_output_arrays(train_df) # 30
    test_bert_inp = du.BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)

    # test
    score_tr = 0
    score_vl = 0
    counter = 0

    fld = MultilabelStratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for ifld, (tr_idxs, vl_idxs) in enumerate(fld.split(train_df.iloc[:,:11], train_df.iloc[:,11:])):
        if ifld == 0:
            #train_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][tr_idxs], train_bert_inp[1][tr_idxs], train_bert_inp[2][tr_idxs], train_bert_inp[3][tr_idxs], train_bert_inp[4][tr_idxs], train_bert_inp[5][tr_idxs], train_bert_label[tr_idxs])
            val_ds = tdu.QADataset_SeparateQA(train_bert_inp[0][vl_idxs], train_bert_inp[1][vl_idxs], train_bert_inp[2][vl_idxs], train_bert_inp[3][vl_idxs], train_bert_inp[4][vl_idxs], train_bert_inp[5][vl_idxs], train_bert_label[vl_idxs])

            train_batch_size = 32
            val_batch_size = 32
            #train_loader = tdu.get_dataloader(train_ds, train_batch_size, shuffle=False)
            val_loader = tdu.get_dataloader(val_ds, val_batch_size, shuffle=False)

            model.load_state_dict(torch.load(model_path[ifld]))

            print('fold {0}'.format(ifld))
            #print('train')
            #score_tr += pu.pred_score_sepQA_1(model, train_loader)
            print('val')
            score_vl += pu.pred_score_sepQA_1(model, val_loader)
            counter += 1

    print('average')
    print('train')
    score_tr = score_tr / counter
    print(score_tr)
    print('val')
    score_vl = score_vl / counter
    print(score_vl)

    return