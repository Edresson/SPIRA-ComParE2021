import os
import math
import torch
import torch.nn as nn
import traceback
import pandas as pd

import time
import numpy as np

import argparse

from utils.generic_utils import load_config, save_config_file
from utils.generic_utils import set_init_dict

from utils.generic_utils import NoamLR, binary_acc

from utils.generic_utils import save_best_checkpoint

from utils.tensorboard import TensorboardWriter

from utils.dataset import test_dataloader

from models.spiraconv import *

from models.panns import *

from utils.audio_processor import AudioProcessor 

from sklearn.metrics import f1_score, recall_score

from utils.models import return_model

from test_all_seeds_or_folds import run_test_all_seeds

import sys
import random
# set random seed
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def run_val_train(experiment_dir, batch_size, num_workers, simples_vote, cuda=True, debug=True, return_potential=False): 
    folds = os.listdir(experiment_dir)
    folds.sort()
    # validation
    uars = []
    f1s = []
    accs_control =[]
    accs_patient = []
    # blockPrint()
    for f in folds:
        fold_dir = os.path.join(experiment_dir, f)
        if os.path.isfile(fold_dir):
            continue
        f1, uar, acc_c, acc_p = run_test_all_seeds(fold_dir, 'eval', '', batch_size, num_workers, simples_vote, '', cuda=True, debug=True, return_potential=False, return_target=False, return_f1_auc=True)
        uars.append(uar)
        f1s.append(f1)
        accs_control.append(acc_c)
        accs_patient.append(acc_p)
    enablePrint()
    # print(uars)
    print("DEVEL:")
    print('Mean UAR:', np.array(uars).mean())
    print('Mean F1:', np.array(f1s).mean())
    print('Mean ACC Control:', np.array(accs_control).mean())
    print('Mean ACC Patient:', np.array(accs_patient).mean())
    # Train
    uars = []
    f1s = []
    accs_control =[]
    accs_patient = []
    # blockPrint()
    for f in folds:
        fold_dir = os.path.join(experiment_dir, f)
        if os.path.isfile(fold_dir):
            continue
        f1, uar, acc_c, acc_p = run_test_all_seeds(fold_dir, 'train', '', batch_size, num_workers, simples_vote, '', cuda=True, debug=True, return_potential=False, return_target=False, return_f1_auc=True)
        uars.append(uar)
        f1s.append(f1)
        accs_control.append(acc_c)
        accs_patient.append(acc_p)

    enablePrint()

    print("Train:")
    print('Mean UAR:', np.array(uars).mean())
    print('Mean F1:', np.array(f1s).mean())
    print('Mean ACC Control:', np.array(accs_control).mean())
    print('Mean ACC Patient:', np.array(accs_patient).mean())

if __name__ == '__main__':
    ''' Rerturn metrics in evaluation and train in Cross validation for darkness experiments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default=None, required=True,
                        help="path of experiment with all seeds or folds")
    parser.add_argument('--batch_size', type=int, default=20,
                        help="Batch size for test")
    parser.add_argument('--num_workers', type=int, default=10,
                        help="Number of Workers for test data load")
    parser.add_argument('--debug', type=bool, default=True,
                        help=" Print classification error")
    parser.add_argument('--simples_vote', type=bool, default=False,
                        help="If True use simple vote, else use composite vote, default False")
                        
    args = parser.parse_args()
    run_val_train(args.experiment_dir, args.batch_size, args.num_workers, args.simples_vote, cuda=True, debug=args.debug, return_potential=False)
    