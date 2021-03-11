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
        
        if args.seed is None:
            log_path = os.path.join(c.train_config['logs_path'], c.model_name)
        else:
            log_path = os.path.join(os.path.join(c.train_config['logs_path'], str(args.seed)), c.model_name)
            c.train_config['seed'] = args.seed

        os.makedirs(log_path, exist_ok=True)

        tensorboard = TensorboardWriter(os.path.join(log_path,'tensorboard'))

        trainloader = train_dataloader(c, ap, class_balancer_batch=c.dataset['class_balancer_batch'])
        max_seq_len = trainloader.dataset.get_max_seq_lenght()
        c.dataset['max_seq_len'] = max_seq_len

        # save config in train dir, its necessary for test before train and reproducity
        save_config_file(c, os.path.join(log_path,'config.json'))

        evaloader = eval_dataloader(c, ap, max_seq_len=max_seq_len)
    
        return train(args, log_path, args.checkpoint_path, trainloader, evaloader, tensorboard, c, c.model_name, ap, cuda=True, model_params=model_params)


     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file, for continue training")
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help="Seed for training")
    args = parser.parse_args()
    c = load_config(args.config_path)
    current_path = c.train_config['logs_path']

    betas = [1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]    

    optimizer = ["adam", "adamw", "radam"]

    weight_decay = [0, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    learning_rate = [1e-3, 1e-2, 1e-4, 1e-5, 1e-6]

    runs = 0


    c = load_config(args.config_path)

    min_loss = 999999

    current_path = c.train_config['logs_path']
    bests = [] 
    enablePrint()
    print("="*20)  
    blockPrint()


    blockPrint()
    enablePrint()
    print("\n\nStart search: betas ")
    blockPrint()
    for p in betas:
        c.train_config['logs_path'] = current_path+'-'+str(runs)
        c_aux = copy_config_dict(c)
        c_aux.model['mixup_alpha'] = p
        loss = run_train(c_aux, args)
        if loss < min_loss:
            c = copy_config_dict(c_aux)
            min_loss = loss
            enablePrint()
            print("="*20)
            print("RUN: ", runs)
            print( "BEST LOSS: ", min_loss)
            print( "Params: ", c)
            print("="*20)
            blockPrint()
            bests.append([min_loss, c, runs])
        runs+= 1
        print("RUN:",runs)

    enablePrint()
    print("\n\nStart search: optimizer ")
    blockPrint()
    for p in optimizer:
        for w in weight_decay:
            c.train_config['logs_path'] = current_path+'-'+str(runs)
            c_aux = copy_config_dict(c)
            c_aux.train_config['weight_decay'] = w
            c_aux.train_config['optimizer'] = p
            loss = run_train(c_aux, args)
            if loss < min_loss:
                c = copy_config_dict(c_aux)
                min_loss = loss
                enablePrint()
                print("="*20)
                print("RUN: ", runs)
                print( "BEST LOSS: ", min_loss)
                print( "Params: ", c)
                print("="*20)
                blockPrint()
                bests.append([min_loss, c, runs])
            runs+= 1
            print("RUN:",runs)

    enablePrint()
    print("\n\nStart search: learning_rate ")
    blockPrint()
    for p in learning_rate:
        c.train_config['logs_path'] = current_path+'-'+str(runs)
        c_aux = copy_config_dict(c)
        c_aux.train_config['learning_rate'] = p
        loss = run_train(c_aux, args)
        if loss < min_loss:
            c = copy_config_dict(c_aux)
            min_loss = loss
            enablePrint()
            print("="*20)
            print("RUN: ", runs)
            print( "BEST LOSS: ", min_loss)
            print( "Params: ", c)
            print("="*20)
            blockPrint()
            bests.append([min_loss, c, runs])
        runs+= 1
        print("RUN:",runs)
    

enablePrint()
print("="*30, 'FINAL', "="*30)
print(bests)
print( "BEST LOSS: ", min_loss)
print( "Params: ", c)