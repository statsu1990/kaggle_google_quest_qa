import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from scipy import stats
from math import floor, ceil
import torch


import config

class InputData:
    @staticmethod
    def get_train_data(to_relative_rank=False, clip_output=None, hard_target_num=None):
        df = pd.read_csv(config.PATH+'train.csv')

        #a = (stats.mstats.rankdata(df.iloc[:,11].values) - 0.5) / len(df.iloc[:,11])
        #for i in range(30):
            #df.iloc[:,11+i] = (stats.mstats.rankdata(df.iloc[:,11+i].values) - 0.5) / len(df.iloc[:,11+i])
            #df.iloc[:,11+i:11+i+1] = (stats.mstats.rankdata(df.iloc[:,11+i:11+i+1].values) - 0.5) / len(df.iloc[:,11+i:11+i+1])
            #df.iloc[:,11+i] = df.iloc[:,11+i]
            #df.iloc[:,11:] = (df.iloc[:,11:].apply(stats.mstats.rankdata, axis=1) - 0.5) / len(df.iloc[:,11+i:11+i+1])
        #b = df.iloc[:,11:].apply(stats.mstats.rankdata, axis=1)
        #c = (np.apply_along_axis(stats.mstats.rankdata, axis=0, arr=df.iloc[:,11:].values) - 0.5) / len(df.iloc[:,11])
        if to_relative_rank:
            df.iloc[:,11:] = (df.iloc[:,11:].apply(stats.mstats.rankdata, axis=0) - 0.5) / len(df.iloc[:,11:])

        if clip_output is not None:
            df.iloc[:,11:] = df.iloc[:,11:].clip(clip_output[0], clip_output[1])

        if hard_target_num is not None:
            s = 0.0
            e = 1.0
            d = (e - s) / hard_target_num

            hard_label = df.iloc[:,11:].values
            for i in range(hard_target_num):
                if i < hard_target_num - 1:
                    hard_label[(df.iloc[:,11:] >= s + i * d) & (df.iloc[:,11:] < s + (i + 1) * d)] = i
                else:
                    hard_label[df.iloc[:,11:] >= s + i * d] = i

            df.iloc[:,11:] = hard_label.astype('int')

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

class BertData:
    def __init__(self,):
        return

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
    def _trim_input(tokenizer, title, question, answer, max_sequence_length, 
                    t_max_len=30, q_max_len=239, a_max_len=239):

        t = tokenizer.tokenize(title)
        q = tokenizer.tokenize(question)
        a = tokenizer.tokenize(answer)
    
        t_len = len(t)
        q_len = len(q)
        a_len = len(a)

        if (t_len+q_len+a_len+4) > max_sequence_length:
        
            if t_max_len > t_len:
                t_new_len = t_len
                a_max_len = a_max_len + floor((t_max_len - t_len)/2)
                q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
            else:
                t_new_len = t_max_len
      
            if a_max_len > a_len:
                a_new_len = a_len 
                q_new_len = q_max_len + (a_max_len - a_len)
            elif q_max_len > q_len:
                a_new_len = a_max_len + (q_max_len - q_len)
                q_new_len = q_len
            else:
                a_new_len = a_max_len
                q_new_len = q_max_len
            
            
            if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
                raise ValueError("New sequence length should be %d, but is %d" 
                                 % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
        
            t = t[:t_new_len]
            q = q[:q_new_len]
            a = a[:a_new_len]
    
        return t, q, a

    @staticmethod
    def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
        """Converts tokenized input to ids, masks and segments for BERT"""
    
        stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

        input_ids = BertData._get_ids(stoken, tokenizer, max_sequence_length)
        input_masks = BertData._get_masks(stoken, max_sequence_length)
        input_segments = BertData._get_segments(stoken, max_sequence_length)

        return [input_ids, input_masks, input_segments]

    @staticmethod
    def get_input_categories(df_train):
        return list(df_train.columns[[1,2,5]])

    @staticmethod
    def get_output_categories(df_train):
        return list(df_train.columns[11:])

    @staticmethod
    def compute_input_arrays(df, tokenizer, max_sequence_length):
        columns = BertData.get_input_categories(df)

        input_ids, input_masks, input_segments = [], [], []
        for _, instance in df[columns].iterrows():
            t, q, a = instance.question_title, instance.question_body, instance.answer

            t, q, a = BertData._trim_input(tokenizer, t, q, a, max_sequence_length)

            ids, masks, segments = BertData._convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
        
        return [torch.tensor(np.asarray(input_ids, dtype=np.int64)), 
                torch.tensor(np.asarray(input_masks, dtype=np.int64)), 
                torch.tensor(np.asarray(input_segments, dtype=np.int64))]

    @staticmethod
    def compute_output_arrays(df):
        columns = BertData.get_output_categories(df)
        return torch.tensor(np.asarray(df[columns]))

class BertData_v2:
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
    def _trim_input(tokenizer, title, question, answer, max_sequence_length, 
                    t_max_len=30, q_max_len=240, a_max_len=239):

        t = tokenizer.tokenize(title)
        q = tokenizer.tokenize(question)
        a = tokenizer.tokenize(answer)
    
        t_len = len(t)
        q_len = len(q)
        a_len = len(a)

        if (t_len+q_len+a_len+BertData_v2.num_special_token) > max_sequence_length:
        
            if t_max_len > t_len:
                t_new_len = t_len
                a_max_len = a_max_len + floor((t_max_len - t_len)/2)
                q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
            else:
                t_new_len = t_max_len
      
            if a_max_len > a_len:
                a_new_len = a_len 
                q_new_len = q_max_len + (a_max_len - a_len)
            elif q_max_len > q_len:
                a_new_len = a_max_len + (q_max_len - q_len)
                q_new_len = q_len
            else:
                a_new_len = a_max_len
                q_new_len = q_max_len
            
            
            if t_new_len+a_new_len+q_new_len+BertData_v2.num_special_token != max_sequence_length:
                raise ValueError("New sequence length should be %d, but is %d" 
                                 % (max_sequence_length, (t_new_len+a_new_len+q_new_len+BertData_v2.num_special_token)))
        
            t = t[:t_new_len]
            q = q[:q_new_len]
            a = a[:a_new_len]
    
        return t, q, a

    @staticmethod
    def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
        """Converts tokenized input to ids, masks and segments for BERT"""
    
        #stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]
        stoken = ["[CLS]"] + title + question + ["[SEP]"] + answer + ["[SEP]"]

        input_ids = BertData_v2._get_ids(stoken, tokenizer, max_sequence_length)
        input_masks = BertData_v2._get_masks(stoken, max_sequence_length)
        input_segments = BertData_v2._get_segments(stoken, max_sequence_length)

        return [input_ids, input_masks, input_segments]

    @staticmethod
    def get_input_categories(df_train):
        return list(df_train.columns[[1,2,5]])

    @staticmethod
    def get_output_categories(df_train):
        return list(df_train.columns[11:])

    @staticmethod
    def compute_input_arrays(df, tokenizer, max_sequence_length):
        columns = BertData_v2.get_input_categories(df)

        input_ids, input_masks, input_segments = [], [], []
        for _, instance in df[columns].iterrows():
            t, q, a = instance.question_title, instance.question_body, instance.answer

            t, q, a = BertData_v2._trim_input(tokenizer, t, q, a, max_sequence_length)

            ids, masks, segments = BertData_v2._convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
        
        return [torch.tensor(np.asarray(input_ids, dtype=np.int64)), 
                torch.tensor(np.asarray(input_masks, dtype=np.int64)), 
                torch.tensor(np.asarray(input_segments, dtype=np.int64))]

    @staticmethod
    def compute_output_arrays(df):
        columns = BertData_v2.get_output_categories(df)
        return torch.tensor(np.asarray(df[columns]))

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
