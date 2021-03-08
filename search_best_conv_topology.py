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

from utils.generic_utils import save_best_checkpoint

from utils.tensorboard import TensorboardWriter

from utils.dataset import train_dataloader, eval_dataloader

from models.spiraconv import *
from utils.audio_processor import AudioProcessor 

from train import train
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def run_train(c, args, model_params):

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

    

    kernels = [(7,1), (5, 1), (3, 1), (2, 1), (3, 3)]
    dilatation = [ (1, 1), (2,1), (3,1), (4,1)]

    fc1_dim = [10, 30, 50, 100, 150, 300, 500, 1000]

    num_layers = [6, 2, 3, 4, 5]

    dropout_rates = [0.2, 0.4, 0.7, 0.8, 0.9]
    


    runs = 0 


    c = load_config(args.config_path)
    min_loss = 999999

    current_path = c.train_config['logs_path']
    bests = [] 
    enablePrint()
    print("="*20)  
    blockPrint()

    best_params = {'config':c, 'dropout_rate':0.7, 'conv_num':4, 'kernels':(3,1), 'dilatation': (2,1)}

    blockPrint()

    enablePrint()
    print("\n\nStart search: num_layers ")
    blockPrint()

    for p in num_layers:
        c.train_config['logs_path'] = current_path+'-'+str(runs)
        temp_parms =  best_params.copy()
        temp_parms['conv_num'] = p
        loss = run_train(c, args, temp_parms)
        if loss < min_loss:
            best_params = temp_parms.copy()
            min_loss = loss
            enablePrint()
            print("="*20)
            print("RUN: ", runs)
            print( "BEST LOSS: ", min_loss)
            print( "Params: ", best_params)
            print("="*20)
            blockPrint()
            bests.append([min_loss, best_params, runs])
        runs+= 1
        print(runs)
    enablePrint()
    print("\n\nStart search: kernels ")
    blockPrint()


    for p in kernels:
        c.train_config['logs_path'] = current_path+'-'+str(runs)
        temp_parms =  best_params.copy()
        temp_parms['kernels'] = p
        loss = run_train(c, args, temp_parms)
        if loss < min_loss:
            best_params = temp_parms.copy()
            min_loss = loss
            enablePrint()
            print("="*20)
            print("RUN: ", runs)
            print( "BEST LOSS: ", min_loss)
            print( "Params: ", best_params)
            print("="*20)
            blockPrint()
            bests.append([min_loss, best_params, runs])
        runs+= 1
        
    enablePrint()
    print("\n\nStart search: dropout_rates ")
    blockPrint()
    for p in dropout_rates:
        c.train_config['logs_path'] = current_path+'-'+str(runs)
        temp_parms =  best_params.copy()
        temp_parms['dropout_rate'] = p
        loss = run_train(c, args, temp_parms)
        if loss < min_loss:
            best_params = temp_parms.copy()
            min_loss = loss
            enablePrint()
            print("="*20)
            print("RUN: ", runs)
            print( "BEST LOSS: ", min_loss)
            print( "Params: ", best_params)
            print("="*20)
            blockPrint()
            bests.append([min_loss, best_params, runs])
        runs+= 1

    enablePrint()



    print("\n\nStart search: dilatation ")
    blockPrint()
    for p in dilatation:
        c.train_config['logs_path'] = current_path+'-'+str(runs)
        temp_parms =  best_params.copy()
        temp_parms['dilatation'] = p
        loss = run_train(c, args, temp_parms)
        if loss < min_loss:
            best_params = temp_parms.copy()
            min_loss = loss
            enablePrint()
            print("="*20)
            print("RUN: ", runs)
            print( "BEST LOSS: ", min_loss)
            print( "Params: ", best_params)
            print("="*20)
            blockPrint()
            bests.append([min_loss, best_params, runs])
        runs+= 1

    enablePrint()
    print("\n\nStart search: config.model['fc1_dim'] ")
    blockPrint()
    for p in fc1_dim:
        c.train_config['logs_path'] = current_path+'-'+str(runs)
        c.model['fc1_dim'] = p

        temp_parms =  best_params.copy()
        temp_parms['config'] = c
        loss = run_train(c, args, temp_parms)
        if loss < min_loss:
            best_params = temp_parms.copy()
            min_loss = loss
            enablePrint()
            print("="*20)
            print("RUN: ", runs)
            print( "BEST LOSS: ", min_loss)
            print( "Params: ", best_params)
            print("="*20)
            blockPrint()
            bests.append([min_loss, best_params, runs])
        runs+= 1

enablePrint()
print("="*30, 'FINAL', "="*30)
print(bests)
print( "BEST LOSS: ", min_loss)
print( "Params: ", best_params)