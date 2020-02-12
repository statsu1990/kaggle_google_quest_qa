from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from scipy import stats
from scipy.stats import spearmanr

from tqdm import tqdm

from math import floor, ceil
import os

class config:
    PATH = '../input/google-quest-challenge/'

    BERT_PRETRAINED_MODEL = 'bert-base-uncased'
    BERT_TOKENIZER_PATH = '../input/transformers/' + BERT_PRETRAINED_MODEL + '/tokenizer/'
    BERT_PRETRAINED_MODEL_PATH = '../input/transformers/' + BERT_PRETRAINED_MODEL + '/pretrained_model/'

    MY_MODEL = '../input/mymodel/20020302_bert_v1_0/'

class InputData:
    @staticmethod
    def get_train_data(to_relative_rank=False, clip_output=None):
        df = pd.read_csv(config.PATH+'train.csv')

        if to_relative_rank:
            df.iloc[:,11:] = (df.iloc[:,11:].apply(stats.mstats.rankdata, axis=0) - 0.5) / len(df.iloc[:,11:])

        if clip_output is not None:
            df.iloc[:,11:] = df.iloc[:,11:].clip(clip_output[0], clip_output[1])

        return df

    @staticmethod
    def get_test_data():
        return pd.read_csv(config.PATH+'test.csv')

class Submission:
    @staticmethod
    def get_submission_file():
        return pd.read_csv(config.PATH+'sample_submission.csv')

    @staticmethod
    def make_submission(pred, filename='submission.csv'):
        df_sub = Submission.get_submission_file()
        df_sub.iloc[:, 1:] = pred
        df_sub.to_csv(filename, index=False)
        return

class BertData_v3:
    num_special_token = 3

    @staticmethod
    def _get_masks(tokens, max_seq_length):
        if len(tokens)>max_seq_length:
            raise IndexError("Token length more than max seq length!")
        return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

    @staticmethod
    def _get_segments(tokens, max_seq_length):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens)>max_seq_length:
            raise IndexError("Token length more than max seq length!")
        segments = []
        first_sep = True
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                if first_sep:
                    first_sep = False 
                else:
                    current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))

    @staticmethod
    def _get_ids(tokens, tokenizer, max_seq_length):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.encode(tokens, add_special_tokens=False)
        input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
        return input_ids

    @staticmethod
    def _trim_input(tokenizer, title, body, max_sequence_length, t_max_len=50):
        t = tokenizer.tokenize(title)
        b = tokenizer.tokenize(body)
    
        t_len = len(t)
        b_len = len(b)

        if (t_len + b_len + BertData_v3.num_special_token) > max_sequence_length:
            diff = (t_len + b_len + BertData_v3.num_special_token) - max_sequence_length

            if t_len > t_max_len:
                t_len_new = max(t_max_len, t_len - diff)
            else:
                t_len_new = t_len

            if (t_len_new + b_len + BertData_v3.num_special_token) > max_sequence_length:
                b_len_new = max_sequence_length - t_len_new - BertData_v3.num_special_token
            else:
                b_len_new = b_len

            t = t[:t_len_new]
            b = b[:b_len_new]

        return t, b

    @staticmethod
    def _convert_to_bert_inputs(title, body, tokenizer, max_sequence_length):
        """Converts tokenized input to ids, masks and segments for BERT"""
    
        stoken = ["[CLS]"] + title + ["[SEP]"] + body + ["[SEP]"]

        input_ids = BertData_v3._get_ids(stoken, tokenizer, max_sequence_length)
        input_masks = BertData_v3._get_masks(stoken, max_sequence_length)
        input_segments = BertData_v3._get_segments(stoken, max_sequence_length)

        return [input_ids, input_masks, input_segments]

    @staticmethod
    def get_input_categories(df_train):
        return list(df_train.columns[[1,2,5]])

    @staticmethod
    def get_output_categories(df_train):
        return list(df_train.columns[11:])

    @staticmethod
    def compute_input_arrays(df, tokenizer, max_sequence_length):
        columns = BertData_v3.get_input_categories(df)

        input_q_ids, input_q_masks, input_q_segments = [], [], []
        input_a_ids, input_a_masks, input_a_segments = [], [], []

        counter = 0
        for _, instance in df[columns].iterrows():
            t, q, a = instance.question_title, instance.question_body, instance.answer

            t_token, q_token = BertData_v3._trim_input(tokenizer, t, q, max_sequence_length)
            ids, masks, segments = BertData_v3._convert_to_bert_inputs(t_token, q_token, tokenizer, max_sequence_length)
            input_q_ids.append(ids)
            input_q_masks.append(masks)
            input_q_segments.append(segments)
            
            t_token, a_token = BertData_v3._trim_input(tokenizer, t, a, max_sequence_length)
            ids, masks, segments = BertData_v3._convert_to_bert_inputs(t_token, a_token, tokenizer, max_sequence_length)
            input_a_ids.append(ids)
            input_a_masks.append(masks)
            input_a_segments.append(segments)
            
        return [
                torch.tensor(np.asarray(input_q_ids, dtype=np.int64)), 
                torch.tensor(np.asarray(input_q_masks, dtype=np.int64)), 
                torch.tensor(np.asarray(input_q_segments, dtype=np.int64)),
                torch.tensor(np.asarray(input_a_ids, dtype=np.int64)), 
                torch.tensor(np.asarray(input_a_masks, dtype=np.int64)), 
                torch.tensor(np.asarray(input_a_segments, dtype=np.int64)),
                ]

    @staticmethod
    def compute_output_arrays(df):
        columns = BertData_v3.get_output_categories(df)
        return torch.tensor(np.asarray(df[columns]))

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

class QADataset_SeparateQA(torch.utils.data.Dataset):
    def __init__(self, q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, labels=None):
        self.q_ids = q_ids
        self.q_masks = q_masks
        self.q_segments = q_segments
        self.a_ids = a_ids
        self.a_masks = a_masks
        self.a_segments = a_segments
        self.labels = labels

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.q_ids[idx], self.q_masks[idx], self.q_segments[idx], self.a_ids[idx], self.a_masks[idx], self.a_segments[idx], self.labels[idx]
        else:
            return self.q_ids[idx], self.q_masks[idx], self.q_segments[idx], self.a_ids[idx], self.a_masks[idx], self.a_segments[idx]

    def __len__(self):
        return len(self.q_ids)

def get_dataloader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def compute_spearmanr(original, preds):
    scores = []
    for i in range(30):
        scores.append(spearmanr(original[:, i], preds[:, i]).correlation)
    print(scores)
    return np.nanmean(scores)

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def calc_pred_Bert_v1_0(net, dataloader, to_relative_rank=False, with_target=False):
    net = net.cuda()
    
    net.eval()

    preds = []
    original = []
    if with_target:
        with torch.no_grad():
            for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments, targets) in enumerate(tqdm(dataloader)):
                q_ids, q_masks, q_segments, targets = q_ids.cuda(), q_masks.cuda(), q_segments.cuda(), targets.cuda()
                a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
                outputs, hidden_outpus = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)

                preds.append(outputs.cpu().numpy())
                original.append(targets.cpu().numpy())
    else:
        with torch.no_grad():
            for batch_idx, (q_ids, q_masks, q_segments, a_ids, a_masks, a_segments) in enumerate(tqdm(dataloader)):
                q_ids, q_masks, q_segments = q_ids.cuda(), q_masks.cuda(), q_segments.cuda()
                a_ids, a_masks, a_segments = a_ids.cuda(), a_masks.cuda(), a_segments.cuda()
                outputs, hidden_outpus = net(q_ids, q_masks, q_segments, a_ids, a_masks, a_segments)

                preds.append(outputs.cpu().numpy())

    preds = np.concatenate(preds)
    preds = sigmoid(preds)

    if to_relative_rank:
        preds = np.apply_along_axis(stats.mstats.rankdata, axis=0, arr=preds) / len(preds)

    if with_target:
        original = np.concatenate(original)    
        score = compute_spearmanr(original, preds)
        print('Score: %.5f' % (score,))

    return preds

def calc_pred_question_type_spelling(data_df):
    # category, host
    ex_x = data_df[data_df['category']=='CULTURE']
    ex_x = ex_x[ex_x['host']=='english.stackexchange.com']

    qts = ex_x['question_title']

    #posi = []
    #for qt in qts:
    #    words = qt.split(' ')
    #    or_flg = []
    #    or_flg.append('pronounce' in words)
    #    or_flg.append('pronunciation' in words)
    #    or_flg.append('sounds' in words)
    #    or_flg.append('sound' in words)
    #    or_flg.append('spell' in words)
    #    or_flg.append('spells' in words)
    #    or_flg.append('spelling' in words)
    #    or_flg.append('accent' in words)
    #    or_flg.append('accented' in words)
    #    or_flg.append('vowel' in words)
    #    or_flg.append('schwa' in words)
    #    or_flg.append('syllables' in words)
    #    or_flg.append('phoneme' in words)
    #    #or_flg.append('or' in words)
    #    #or_flg.append('and' in words)
    #        
    #    and_flg = []
    #    and_flg.append(not 'mean' in words)
    #    and_flg.append(not 'means' in words)
    #    and_flg.append(not 'meaning' in words)
    #
    #    if any(or_flg) and all(and_flg):
    #        posi.append(1)
    #        print(qt)
    #    else:
    #        posi.append(0)
    #posi = np.array(posi)
    #if np.all(posi < 0.5):
    #    posi[0] = 1
    
    pred = np.zeros(len(data_df))
    #pred[ex_x.index] = posi
    pred[ex_x.index] = 1
    
    return pred

def run_20020401():
    DO_CHECK = True

    print('model')
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    model_path = [
        os.path.join(config.MY_MODEL, 'Bert_v1_0_model_fold0'),
        os.path.join(config.MY_MODEL, 'Bert_v1_0_model_fold1'),
        os.path.join(config.MY_MODEL, 'Bert_v1_0_model_fold2'),
        os.path.join(config.MY_MODEL, 'Bert_v1_0_model_fold3'),
        os.path.join(config.MY_MODEL, 'Bert_v1_0_model_fold4'),
        ]

    model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

    print('test data')
    # raw data
    test_df = InputData.get_test_data()
    # bert input
    test_bert_inp = BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)
    # prediction
    pred = None
    counter = 0
    for mp in model_path:
        ds = QADataset_SeparateQA(test_bert_inp[0], test_bert_inp[1], test_bert_inp[2], 
                                  test_bert_inp[3], test_bert_inp[4], test_bert_inp[5])

        batch_size = 32
        loader = get_dataloader(ds, batch_size, shuffle=False)

        model.load_state_dict(torch.load(mp))

        if pred is None:
            pred = calc_pred_Bert_v1_0(model, loader, with_target=False)
        else:
            pred = pred + calc_pred_Bert_v1_0(model, loader, with_target=False)
        counter += 1
    pred = pred / counter
    # make submission
    print('submission')
    Submission.make_submission(pred)

    if DO_CHECK:
        print('train data')
        # raw data
        train_df = InputData.get_train_data()
        # bert input
        train_bert_inp = BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
        train_bert_label = BertData_v3.compute_output_arrays(train_df) # 30
        # prediction
        pred = None
        counter = 0
        for mp in model_path:
            ds = QADataset_SeparateQA(train_bert_inp[0], train_bert_inp[1], train_bert_inp[2], 
                                      train_bert_inp[3], train_bert_inp[4], train_bert_inp[5],
                                      train_bert_label)

            batch_size = 32
            loader = get_dataloader(ds, batch_size, shuffle=False)

            model.load_state_dict(torch.load(mp))

            if pred is None:
                pred = calc_pred_Bert_v1_0(model, loader, with_target=True)
            else:
                pred = pred + calc_pred_Bert_v1_0(model, loader, with_target=True)
            counter += 1
        pred = pred / counter
        # make submission
        score = compute_spearmanr(train_bert_label, pred)
        print('train data score : {0}'.format(score))

    return

def run_20020601():
    DO_CHECK = True

    print('model')
    pretrained_model = config.BERT_PRETRAINED_MODEL
    pretrained_tokenizer_path = config.BERT_TOKENIZER_PATH
    pretrained_model_path = config.BERT_PRETRAINED_MODEL_PATH
    max_sequence_length = 512

    tokenizer = BertUtils.get_bert_tokenizer(pretrained_tokenizer_path)
    
    model_path = [
        os.path.join(config.MY_MODEL, 'Bert_v1_0_model_fold0'),
        #os.path.join(config.MY_MODEL, 'Bert_v1_0_model_fold1'),
        #os.path.join(config.MY_MODEL, 'Bert_v1_0_model_fold2'),
        #os.path.join(config.MY_MODEL, 'Bert_v1_0_model_fold3'),
        #os.path.join(config.MY_MODEL, 'Bert_v1_0_model_fold4'),
        ]

    model = Bert_v1_0(BertUtils.get_bert_model(pretrained_model_path))

    print('test data')
    # raw data
    test_df = InputData.get_test_data()
    # bert input
    test_bert_inp = BertData_v3.compute_input_arrays(test_df, tokenizer, max_sequence_length)
    # prediction
    pred = None
    counter = 0
    for mp in model_path:
        ds = QADataset_SeparateQA(test_bert_inp[0], test_bert_inp[1], test_bert_inp[2], 
                                  test_bert_inp[3], test_bert_inp[4], test_bert_inp[5])

        batch_size = 32
        loader = get_dataloader(ds, batch_size, shuffle=False)

        model.load_state_dict(torch.load(mp))

        if pred is None:
            pred = calc_pred_Bert_v1_0(model, loader, with_target=False)
        else:
            pred = pred + calc_pred_Bert_v1_0(model, loader, with_target=False)
        counter += 1
    pred = pred / counter
    # predcition by rule base
    pred_question_type_spelling = calc_pred_question_type_spelling(test_df)
    pred[:,19] = pred_question_type_spelling

    # make submission
    print('submission')
    Submission.make_submission(pred)

    if DO_CHECK:
        print('train data')
        # raw data
        train_df = InputData.get_train_data()
        # bert input
        train_bert_inp = BertData_v3.compute_input_arrays(train_df, tokenizer, max_sequence_length)
        train_bert_label = BertData_v3.compute_output_arrays(train_df) # 30
        # prediction
        pred = None
        counter = 0
        for mp in model_path:
            ds = QADataset_SeparateQA(train_bert_inp[0], train_bert_inp[1], train_bert_inp[2], 
                                      train_bert_inp[3], train_bert_inp[4], train_bert_inp[5],
                                      train_bert_label)

            batch_size = 32
            loader = get_dataloader(ds, batch_size, shuffle=False)

            model.load_state_dict(torch.load(mp))

            if pred is None:
                pred = calc_pred_Bert_v1_0(model, loader, with_target=True)
            else:
                pred = pred + calc_pred_Bert_v1_0(model, loader, with_target=True)
            counter += 1
        pred = pred / counter
        # make submission
        score = compute_spearmanr(train_bert_label, pred)
        print('train data score : {0}'.format(score))

        # predcition by rule base
        pred_question_type_spelling = calc_pred_question_type_spelling(train_df)
        pred[:,19] = pred_question_type_spelling
        # make submission
        score = compute_spearmanr(train_bert_label, pred)
        print('train data score : {0}'.format(score))

    return