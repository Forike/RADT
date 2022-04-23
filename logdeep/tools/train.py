#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import sys
import time
sys.path.append('../../')
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from logdeep.tools.utils import (save_parameters)
from torch import optim
from logdeep.tools.Optim import ScheduledOptim
from logdeep.tools.Transformer import Transformer
from collections import defaultdict
import itertools

def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict

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



class Trainer():
    def __init__(self, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.data_dir = options['data_dir']
        self.D_1_path = options['D_1_path']
        self.vector_path = options['vector_path']
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        self.device = options['device']
        self.max_epoch = options['max_epoch']
        self.n_warmup_steps = options['n_warmup_steps']
        self.lr_max = options['lr_max']
        self.decay_rate = options['decay_rate']
        self.decay_steps = options['decay_steps']


        self.sample_ratio_D_1 = options["sample_ratio_D_1"]
        self.adam_betas = options["adam_betas"]

        self.d_model = options['d_model']
        self.d_inner = options['d_inner']
        self.n_layers = options['n_layers']
        self.n_head = options['n_head']
        self.d_k = options['d_k']
        self.d_v = options['d_v']
        self.dropout = options['dropout']
        self.n_position = options['n_position']
        os.makedirs(self.save_dir, exist_ok=True)

        data_path=self.data_dir+self.D_1_path
        vector_path=self.data_dir+self.vector_path
        D_1=Logs(data_path=data_path, vector_path=vector_path,
                           window_size=self.window_size, sample_ratio=self.sample_ratio_D_1)
        self.D_1_loader = DataLoader(D_1,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     pin_memory=True)

        print('Train batch size %d' %(options['batch_size']))

        self.model = Transformer(d_model=self.d_model,d_inner=self.d_inner,n_layers=self.n_layers,
                                 n_head=self.n_head,d_k=self.d_k,d_v=self.d_v,dropout=self.dropout,
                                 n_position=self.n_position,device=self.device)

        self.model.to(self.device)

        if options['optimizer'] == 'adam':
            self.optimizer = ScheduledOptim(
                optim.Adam(self.model.parameters(), betas=self.adam_betas, eps=1e-09),
                self.n_warmup_steps, self.lr_max, self.decay_rate, self.decay_steps)

        else:
            raise NotImplementedError

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        save_parameters(options, self.save_dir + "parameters.txt")
        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
        }
        if options['resume_path'] is not None:   #Following the last training
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.optimizer.load_n_steps(checkpoint['n_steps'])


    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
            checkpoint['n_steps'] = self.optimizer.n_steps
        save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()
        criterion = nn.MSELoss()
        tbar = tqdm(self.D_1_loader, desc="\r")
        num_batch = len(self.D_1_loader)
        total_losses = 0
        for i, (session_idx,enc_input, dec_input, dec_output, length) in enumerate(tbar):
            self.optimizer.update_learning_rate()
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            enc_input,dec_input,dec_output = enc_input.to(self.device),dec_input.to(self.device),dec_output.to(self.device)
            enc_input_l,dec_input_l=get_length(length, window_size=self.window_size)
            enc_input_l,dec_input_l=enc_input_l.to(self.device),dec_input_l.to(self.device)
            output,*_ = self.model(enc_input, dec_input, enc_input_l, dec_input_l)
            output = output.reshape(enc_input.size(0),-1,self.d_model)
            loss = criterion(output, dec_output)
            total_losses += float(loss)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.8f (lr: %.6f)" % (total_losses / (i + 1),lr))

        self.log['train']['loss'].append(total_losses / num_batch)

    def start_train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            self.train(epoch)

            if epoch % 5 == 0:
                self.save_checkpoint(epoch,
                                     save_optimizer=True,
                                     suffix="epoch" + str(epoch))
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            self.save_log()
