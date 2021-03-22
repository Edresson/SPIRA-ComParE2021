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

from test_all_seeds_or_folds import *

import sys
import random
# set random seed
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def run_ensamble_multi_exp(experiment_dirs, test_csv, test_root_dir, batch_size, num_workers, simples_vote, output_csv, cuda=True, debug=False, return_potential=False, insert_noise=False, num_additive_noise=0, num_specaug=0, noisetypes=None, musan_path=None):
    votes = []
    wav_files = []
    targets = []
    if debug:
        return_target = True
    else:
        return_target = False
    out_name = ''
    for exp in experiment_dirs:
        if out_name != '':
            out_name += '_' + os.path.basename(os.path.normpath(exp))
        else:
            out_name += os.path.basename(os.path.normpath(exp))
        vote, target, wav_paths, c = run_test_all_seeds_folds(exp, test_csv, test_root_dir, batch_size, num_workers, simples_vote, output_csv, cuda=True, debug=False, return_potential=True, return_target=return_target,  insert_noise=insert_noise, num_additive_noise=num_additive_noise, num_specaug=num_specaug, noisetypes=noisetypes, musan_path=musan_path)
        targets.append(target)
        wav_files.append(wav_paths)
        # process each classify result
        if simples_vote:
            vote = np.array(votes).round()
        votes.append(vote)

        # integrity check
        if len(wav_files):
            if wav_files[-1] != wav_files[0]:
                raise ValueError("Diferents files  or order for the test in diferrents seeds or folds")
    
    # mean vote, and round is necessary if use composite vote
    enablePrint()
    preds = np.mean(np.array(votes), axis=0)
    # print(preds)
    if not return_potential:
        preds = preds.round()
    file_names = wav_files[0]
    if len(targets):
       targets = targets[0]
    if debug and not return_potential:
        enablePrint()
        targets = np.array(targets)
        preds = np.array(preds)
        names = np.array(file_names)
        idxs = np.nonzero(targets == c.dataset['control_class'])
        control_target = targets[idxs]
        control_preds = preds[idxs]
        names_control = names[idxs]

        idxs = np.nonzero(targets == c.dataset['patient_class'])
        
        patient_target = targets[idxs]
        patient_preds = preds[idxs]
        names_patient = names[idxs]

        if debug:
            print('+'*40)
            print("Control Files Classified incorrectly:")
            incorrect_ids = np.nonzero(control_preds != c.dataset['control_class'])
            inc_names = names_control[incorrect_ids]
            print("Num. Files:", len(inc_names))
            print(inc_names)
            print('+'*40)
            print('-'*40)
            print("Patient Files Classified incorrectly:")
            incorrect_ids = np.nonzero(patient_preds != c.dataset['patient_class'])
            inc_names = names_patient[incorrect_ids]
            print("Num. Files:", len(inc_names))
            print(inc_names)
            print('-'*40)

        
        acc_control = (control_preds == control_target).mean()
        acc_patient = (patient_preds == patient_target).mean()
        acc_balanced = (acc_control + acc_patient) / 2

        f1 = f1_score(targets.tolist(), preds.tolist())
        uar = recall_score(targets.tolist(), preds.tolist(), average='macro')
        print("======== Confusion Matrix ==========")
        y_target = pd.Series(targets, name='Target')
        y_pred = pd.Series(preds, name='Predicted')
        df_confusion = pd.crosstab(y_target, y_pred, rownames=['Target'], colnames=['Predicted'], margins=True)
        print(df_confusion)
            
        print("Test\n ", "Acurracy Control: ", acc_control, "Acurracy Patient: ", acc_patient, "Acurracy Balanced", acc_balanced)
        print("F1:", f1, "UAR:", uar)
    
    if return_potential:
        return preds, targets, file_names, c 
    else:
        df = pd.DataFrame({'filename': file_names, 'prediction':preds.astype(int)})
        df['prediction'] = df['prediction'].replace(int(c.dataset['control_class']), 'negative', regex=True).replace(int(c.dataset['patient_class']), 'positive', regex=True)
        if output_csv:
            out_csv_path = output_csv
        else:
            print(out_name)
            out_csv_path = os.path.join(experiment_dirs[0], '..', out_name+'_'+os.path.basename(c.dataset['test_csv']))
        print("Output saved in:",  out_csv_path)
        df.to_csv(out_csv_path, index=False)


if __name__ == '__main__':
    # python test_experiments_ensambles_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_normalized  --batch_size 30

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_csv', type=str, required=True,
                        help="test csv example: ../SPIRA_Dataset_V1/metadata_test.csv")
    parser.add_argument('-r', '--test_root_dir', type=str, required=True,
                        help="Test root dir example: ../SPIRA_Dataset_V1/")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for test")
    parser.add_argument('--num_workers', type=int, default=10,
                        help="Number of Workers for test data load")
    parser.add_argument('--debug', type=int, default=1,
                        help=" Print classification error")
    parser.add_argument('--output_csv', type=str, default=None,
                        help="CSV output")
    parser.add_argument('--insert_noise', type=int, default=0,
                        help=" Insert Noise in test mode")
    parser.add_argument('--num_additive', type=int, default=0,
                        help=" Number of additive noise, default 0 (desable)")
    parser.add_argument('--num_specaug', type=int, default=0,
                        help=" Number of SpecAug noise, default 0 (desable)")                    
                        
    parser.add_argument("--noisetypes", type=list, default=["noise"],
                        help="Musan noise types, default noise")

    parser.add_argument("--musan_path", type=str, default="../musan/",
                        help="Musan dataset Path, default ../musan/")                    
             
    args = parser.parse_args()

    # experiments_dir = ["../Speech/Experiments_Final/Experiment-1/", "../Speech/Experiments_Final/Experiment-2/", "../Speech/Experiments_Final/Experiment-3/", "../Speech/Experiments_Final/Experiment-4/" ]
    experiments_dir = ["../Speech/Experiments_Final/Experiment-3/", "../Speech/Experiments_Final_kfolds/Experiment-2/" ]
    # experiments_dir = ["../Tosse/Experiments_Final/Experiment-1/", "../Tosse/Experiments_Final/Experiment-2/", "../Tosse/Experiments_Final/Experiment-3/", "../Tosse/Experiments_Final/Experiment-4/" ]

    
    run_ensamble_multi_exp(experiments_dir, args.test_csv, args.test_root_dir, args.batch_size, args.num_workers, False, args.output_csv, cuda=True, debug=args.debug, return_potential=False, insert_noise=args.insert_noise, num_additive_noise=args.num_additive, num_specaug=args.num_specaug, noisetypes=args.noisetypes, musan_path=args.musan_path) 
    