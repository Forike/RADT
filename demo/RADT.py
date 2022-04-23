#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *

# Config Parameters

options = dict()

options['data_dir'] = '../dataset/0.00/'
options['D_1_path'] = 'hdfs_train_1.txt'
options['test_abnormal_path'] = 'hdfs_test_abnormal.txt'
options['test_normal_path'] = 'hdfs_test_normal.txt'
options['D_2_path'] = 'hdfs_train_2.txt'
options['vector_path'] = 'HDFS_semantic_vec.json'

options['device'] = "cuda"

# Smaple
options['window_size'] = 10  # if fix_window
options["sample_ratio_D_1"]=1
options["sample_ratio_D_2"]=1
options["sample_ratio_test"]=1
# Model
options['input_size'] = 300   #dim(template vector)


options['d_model'] = options['input_size']  # Embedding Size
options['d_inner'] = 2048  # FeedForward dimension
options['n_layers'] = 1  # number of Encoder's and Decoder's Layer
options['n_head'] = 8  # number of heads in Multi-Head Attention
options['d_k'] = 64
options['d_v'] = 64
options['dropout'] = 0.1
options['n_position'] = 500

# Train
options['batch_size'] = 256
options['max_epoch'] = 1  #RADT convergence can be made by using only 1 epoch
options['n_warmup_steps'] = 5000
options['lr_max'] = 0.001
options['decay_rate'] = 0.9
options['decay_steps'] = 1000
options['optimizer'] = 'adam'
options["adam_betas"] = (0.9,0.98)

options['resume_path'] = None
options['model_name'] = "RADT"

# Predict
options['delta_threshold'] = 1
options['gamma_percentile'] = 99.999  #Determine the threshold gamma

def train():
    trainer = Trainer(options)
    trainer.start_train()


def predict(suffix = ""):
    predicter = Predicter(options)
    predicter.predict_unsupervised(suffix)


if __name__ == "__main__":
    options['seed'] = 3
    seed_everything(seed=options['seed'])
    options['save_dir'] = r"../result/RADT/seed=%d/"%options['seed']
    train()
    options['model_path'] = options['save_dir']+"RADT_last.pth"
    predict()

