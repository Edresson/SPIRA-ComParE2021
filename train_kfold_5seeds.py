import os
import math
import torch
import torch.nn as nn
import traceback

import time
import numpy as np

import argparse

from utils.generic_utils import load_config, save_config_file
from utils.generic_utils import set_init_dict

from utils.generic_utils import NoamLR, binary_acc

from utils.generic_utils import save_best_checkpoint, copy_config_dict

from utils.tensorboard import TensorboardWriter

from utils.dataset import train_dataloader, eval_dataloader

from models.spiraconv import *
from models.panns import *

from utils.audio_processor import AudioProcessor 

from train import train
import sys, os
import copy

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def run_train(c, args, model_params=None):

        ap = AudioProcessor(**c.audio)

        log_path = os.path.join(c.train_config['logs_path'], c.model_name)
    

        os.makedirs(log_path, exist_ok=True)

        tensorboard = TensorboardWriter(os.path.join(log_path,'tensorboard'))
        print(c.dataset['train_csv'], c.dataset['eval_csv'])

        trainloader = train_dataloader(c, ap, class_balancer_batch=c.dataset['class_balancer_batch'])
        max_seq_len = trainloader.dataset.get_max_seq_lenght()
        c.dataset['max_seq_len'] = max_seq_len

        print(c.dataset['train_csv'], c.dataset['eval_csv'])
        
        # save config in train dir, its necessary for test before train and reproducity
        save_config_file(c, os.path.join(log_path,'config.json'))

        evaloader = eval_dataloader(c, ap, max_seq_len=max_seq_len)
    
        return train(args, log_path, args.checkpoint_path, trainloader, evaloader, tensorboard, c, c.model_name, ap, cuda=True, model_params=model_params)


     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('-f', '--folds_path', type=str, required=True,
                        help="Path of the Folds")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file, for continue training")
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help="Seed for training")
    args = parser.parse_args()
    c = load_config(args.config_path)
    current_path = c.train_config['logs_path']

    seeds = [42, 43, 44, 45, 46]

    folds = os.listdir(args.folds_path)
    folds.sort()

    c = load_config(args.config_path)

    current_path = c.train_config['logs_path']
    for f in folds:
        c.dataset['train_csv'] = os.path.join(args.folds_path, f, 'train.csv')
        c.dataset['eval_csv'] = os.path.join(args.folds_path, f, 'devel.csv')
        for s in seeds:
            c.train_config['logs_path'] = os.path.join(current_path, str(f), str(s))
            c.train_config['seed'] = s
            loss = run_train(c, args)
            print('-'*30)
            print("FOLD:", f, "SEED:", s, "Best Loss:", loss)
            print('_'*30)