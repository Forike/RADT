#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import pickle
import random
import sys
import time
import pandas as pd
from logdeep.tools.Transformer import Transformer
sys.path.append('../../')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from collections import defaultdict

def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict

class MyOut():
    def __init__(self,filename):
        self.filename = filename
    def print_and_write_to_txt(self,content):
        print(content)
        with open(self.filename,'a+') as file:
            file.write(content+'\n')

class Logs(Dataset):
    def __init__(self,data_path, vector_path,
                 window_size=None, sample_ratio=1, pad_idx=0):
        self.event2semantic_vec = read_json(vector_path)
        self.d_vec = len(list(self.event2semantic_vec.values())[0])
        self.sessions = defaultdict(list)
        with open(data_path,'r') as file:
            textdata = list(file.readlines())
            total_num=len(textdata)
            if sample_ratio < 1:  # down sampling
                sample_num = int(sample_ratio * total_num)
                textdata = random.sample(textdata, sample_num)
            for session_idx, session in tqdm(enumerate(textdata)):
                session = list(map(int, session.strip().split()))
                windows = []  # Records multiple sliding windows in a session
                for j in range(len(session) - window_size + 1):
                    window = session[j:j + window_size]
                    length = len(window)
                    windows.append((session_idx, window, length))
                if not windows:
                    length = len(session)
                    window = (session + [pad_idx] * (window_size - len(session)))
                    windows.append((session_idx, window, length))
                self.sessions[session_idx] = windows
            self.sessions = list(
                itertools.chain(*[v for k, v in self.sessions.items()])
            )
        self.start_vec = [1.]*self.d_vec  #start
        self.end_vec = [-1.]*self.d_vec  #end
        self.event2semantic_vec['0'] = [0.]*self.d_vec  #pad

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        Semantic_pattern=[]
        session_idx,window,length=self.sessions[idx]
        for event in window:
            Semantic_pattern.append(self.event2semantic_vec[str(event)])
        enc_input=torch.tensor(data=Semantic_pattern,dtype=torch.float)

        Semantic_pattern.insert(0,self.start_vec)
        dec_input=torch.tensor(data=Semantic_pattern,dtype=torch.float)

        Semantic_pattern.insert(length + 1,self.end_vec)
        dec_output=torch.tensor(data=Semantic_pattern[1:],dtype=torch.float)
        length = torch.tensor(length,dtype=torch.long)
        return session_idx,enc_input, dec_input, dec_output, length

def get_length(length, window_size):
    batch_size=len(length)
    enc_inputs_l=np.empty(shape=(batch_size, window_size))
    dec_inputs_l=np.empty(shape=(batch_size, window_size + 1))
    for i in range(batch_size):
        enc_inputs_l[i] = [1]*length[i]+[0]*(window_size - length[i])
        dec_inputs_l[i] = [1]*(length[i]+1)+[0]*(window_size - length[i])
    return torch.tensor(enc_inputs_l,dtype=torch.long),torch.tensor(dec_inputs_l,dtype=torch.long)


class Predicter():
    def __init__(self, options):
        self.data_dir = options['data_dir']
        self.test_abnormal_path = options['test_abnormal_path']
        self.test_normal_path = options['test_normal_path']
        self.D_2_path = options['D_2_path']
        self.vector_path = options['vector_path']

        self.device = options['device']
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.input_size = options['input_size']
        self.batch_size = options['batch_size']
        self.sample_ratio_test = options["sample_ratio_test"]
        self.sample_ratio_D_2 = options['sample_ratio_D_2']
        self.save_dir = options['save_dir']

        self.d_model = options['d_model']
        self.d_inner = options['d_inner']
        self.n_layers = options['n_layers']
        self.n_head = options['n_head']
        self.d_k = options['d_k']
        self.d_v = options['d_v']
        self.dropout = options['dropout']
        self.n_position = options['n_position']


        self.delta_threshold = options['delta_threshold']
        self.gamma_percentile = options['gamma_percentile']

        vector_path=self.data_dir+self.vector_path
        data_path = self.data_dir + self.D_2_path
        D_2=Logs(data_path=data_path,vector_path=vector_path,
                           window_size=self.window_size,sample_ratio=self.sample_ratio_D_2)
        self.D_2_loader = DataLoader(D_2,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     pin_memory=True)

        data_path = self.data_dir + self.test_normal_path
        normal_dataset=Logs(data_path=data_path,vector_path=vector_path,
                           window_size=self.window_size,sample_ratio=self.sample_ratio_test)
        self.test_normal_loader = DataLoader(normal_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        data_path = self.data_dir + self.test_abnormal_path
        abnormal_dataset = Logs(data_path=data_path, vector_path=vector_path,
                           window_size=self.window_size,sample_ratio=self.sample_ratio_test)
        self.test_abnormal_loader = DataLoader(abnormal_dataset,
                                             batch_size=self.batch_size,
                                             shuffle=False,
                                             pin_memory=True)

        self.model = Transformer(d_model=self.d_model, d_inner=self.d_inner, n_layers=self.n_layers,
                                 n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, dropout=self.dropout,
                                 n_position=self.n_position, device=self.device)
        self.model.to(self.device)
        self.myOut = MyOut(options["save_dir"]+"record.txt")

    def compute_loss(self,loader,criterion):
        with torch.no_grad():
            LOSS = []
            session_idxs = []
            tbar = tqdm(loader, desc="\r")
            for i, (session_idx,enc_input, dec_input, dec_output, length) in enumerate(
                    tbar):
                enc_input, dec_input, dec_output = enc_input.to(self.device), dec_input.to(self.device), dec_output.to(
                    self.device)
                enc_input_l, dec_input_l = get_length(length, window_size=self.window_size)
                enc_input_l, dec_input_l = enc_input_l.to(self.device), dec_input_l.to(self.device)
                output, *_ = self.model(enc_input, dec_input, enc_input_l,
                                        dec_input_l)  # [batch_size*(window_size+1),d_model]
                output = output.reshape(enc_input.size(0), -1, self.d_model)
                dec_output = dec_output.reshape(enc_input.size(0), -1, self.d_model)
                loss = criterion(output, dec_output)  # [batch_size,window_size+1,d_model]
                loss = loss.mean(dim=-1)  # [batch_size,window_size+1]
                LOSS.extend(loss.cpu().numpy())
                session_idxs.extend(session_idx.cpu().numpy())
            LOSS = np.array(LOSS)
            session_idxs = np.array(session_idxs)
        return LOSS,session_idxs

    def predict_unsupervised(self,suffix = ""):
        self.model.load_state_dict(torch.load(self.model_path)['state_dict'])
        self.model.eval()
        self.myOut.print_and_write_to_txt('model_path: {}'.format(self.model_path))

        criterion=nn.MSELoss(reduction='none')

        DELTAS = {}
        LOSS,session_idxs = self.compute_loss(self.D_2_loader, criterion)
        # with open(self.save_dir+"D_2 predict_%s.pkl"%(suffix),"wb") as file:
        #     pickle.dump(obj=(LOSS,session_idxs),file=file)
        self.myOut.print_and_write_to_txt("D_2 max loss:%.8f"%LOSS.reshape(-1).max())
        self.myOut.print_and_write_to_txt("D_2 mean loss:%.8f"%LOSS.reshape(-1).mean())
        self.myOut.print_and_write_to_txt("D_2 min loss:%.8f"%LOSS.reshape(-1).min())

        # The higher the gamma_percentile, the higher the precision, but the lower the recall
        self.gamma_threshold = np.percentile(a=LOSS.reshape(-1),q=self.gamma_percentile)
        self.myOut.print_and_write_to_txt("gamma_threshold:%.8f"%self.gamma_threshold)

        start_time=time.time()
        LOSS,session_idxs = self.compute_loss(self.test_normal_loader, criterion=criterion)
        # with open(self.save_dir+"normal predict_%s.pkl"%(suffix),"wb") as file:
        #     pickle.dump(obj=(LOSS,session_idxs),file=file)
        predict=LOSS>self.gamma_threshold  # [num_all,window_size+1]
        deltas=predict.sum(axis=1)
        predict = np.stack([session_idxs,deltas], axis=1)
        predict = pd.DataFrame(data=predict, columns=["session_idx", "delta"])
        predict = predict.groupby("session_idx").sum()
        # predict.to_csv(self.save_dir+"normal predict.csv",index=False)
        test_normal_num = predict.shape[0]
        #deltas: Each element represents the number of moments when the sample error is greater than the threshold gamma
        deltas = predict["delta"].values
        FP = (deltas >= self.delta_threshold).sum()  #delta_threshold is often equal to 1
        DELTAS["normal"] = deltas

        self.myOut.print_and_write_to_txt("mean delta of normal:%.8f"%deltas.mean())
        self.myOut.print_and_write_to_txt("normal max loss:%.8f" % LOSS.reshape(-1).mean())
        self.myOut.print_and_write_to_txt("normal mean loss:%.8f"%LOSS.reshape(-1).mean())
        self.myOut.print_and_write_to_txt("normal min loss:%.8f"%LOSS.reshape(-1).min())
        self.myOut.print_and_write_to_txt("the number of the normal samples:%d"%test_normal_num)

        LOSS,session_idxs = self.compute_loss(self.test_abnormal_loader, criterion=criterion)
        # with open(self.save_dir+"abnormal predict_%s.pkl"%(suffix),"wb") as file:
        #     pickle.dump(obj=(LOSS,session_idxs),file=file)
        predict = LOSS > self.gamma_threshold  # [num_all,window_size+1]
        deltas = predict.sum(axis=1)  # [num_all]
        predict = np.stack([session_idxs,deltas], axis=1)
        predict = pd.DataFrame(data=predict, columns=["session_idx", "delta"])
        predict = predict.groupby("session_idx").sum()
        # predict.to_csv(self.save_dir+"abnormal predict.csv",index=False)
        test_abnormal_num = predict.shape[0]
        #deltas: Each element represents the number of moments when the sample error is greater than the threshold gamma
        deltas = predict["delta"].values
        TP = (deltas >= self.delta_threshold).sum()
        DELTAS["abnormal"] = deltas

        self.myOut.print_and_write_to_txt("mean delta of abnormal:%.8f"%deltas.mean())
        self.myOut.print_and_write_to_txt("abnormal max loss:%.8f" % LOSS.reshape(-1).mean())
        self.myOut.print_and_write_to_txt("abnormal mean loss:%.8f"%LOSS.reshape(-1).mean())
        self.myOut.print_and_write_to_txt("abnormal min loss:%.8f"%LOSS.reshape(-1).min())
        self.myOut.print_and_write_to_txt("the number of the abnormal samples:%d"%test_abnormal_num)
        self.myOut.print_and_write_to_txt("test_abnormal_num:%d"%test_abnormal_num)
        FN = test_abnormal_num - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        self.myOut.print_and_write_to_txt('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        self.myOut.print_and_write_to_txt('Finished Predicting')

        elapsed_time = time.time() - start_time
        self.myOut.print_and_write_to_txt('elapsed_time: {}'.format(elapsed_time))

