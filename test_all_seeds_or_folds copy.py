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

import sys

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


import random
# set random seed
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
def test(criterion, ap, model, c, testloader,  cuda, confusion_matrix=False, debug=False, simples_vote=False):
    model.zero_grad()
    model.eval()
    preds = []
    targets = []
    names = []
    with torch.no_grad():
        for feature, target, slices, targets_org, name in testloader:       
            #try:
            if cuda:
                feature = feature.cuda()
                if debug:
                    target = target.cuda()
            output = model(feature).float()

            # output = torch.round(output * 10**4) / (10**4)

            # Calculate loss
            if c.dataset["temporal_control"] == "overlapping":
                # unpack overlapping for calculation loss and accuracy 
                if slices is not None and slices is not None:
                    idx = 0
                    new_output = []
                    new_target = []
                    for i in range(slices.size(0)):
                        num_samples = int(slices[i].cpu().numpy())

                        samples_output = output[idx:idx+num_samples]
                        output_mean = samples_output.mean()
                        if debug: 
                            samples_target = target[idx:idx+num_samples]
                            target_mean = samples_target.mean()
                            new_target.append(target_mean)
                        new_output.append(output_mean)
                        idx += num_samples
                    if debug:
                        target = torch.stack(new_target, dim=0)
                    output = torch.stack(new_output, dim=0)
                    #print(target, targets_org)
                    if cuda:
                        output = output.cuda()
                        if debug:
                            target = target.cuda()
                            targets_org = targets_org.cuda()
                            if not torch.equal(targets_org, target):
                                raise RuntimeError("Integrity problem during the unpack of the overlay for the calculation of accuracy and loss. Check the dataloader !!")

            
            
            
            # calculate binnary accuracy
            if simples_vote:
                y_pred_tag = torch.round(output)
            else:
                y_pred_tag = output
            # print(y_pred_tag.reshape(-1).cpu().numpy().tolist())
            # print(target.reshape(-1).int().cpu().numpy().tolist())
            # exit()
            preds += y_pred_tag.reshape(-1).cpu().numpy().tolist()
            if debug:
                targets += target.reshape(-1).int().cpu().numpy().tolist()

            names += name
    
    return preds, targets, names


            
       


def run_test_all_seeds(args, cuda=True, debug=False, return_potential=False):
    runs_list = os.listdir(args.experiment_dir)
    runs_list.sort()
    num_runs = len(runs_list)

    votes = []
    wav_files = []
    targets = []
    # define loss function
    criterion = nn.BCELoss(reduction='sum')
    for run in runs_list:
        blockPrint()
        run_dir = os.path.join(args.experiment_dir, run)
        if os.path.isfile(run_dir):
            continue
        model_name = os.listdir(run_dir)[0]
        checkpoint_path = os.path.join(run_dir, model_name, 'best_checkpoint.pt')
        config_path = os.path.join(run_dir, model_name, 'config.json')

        c = load_config(config_path)
        ap = AudioProcessor(**c.audio)

        c.dataset['test_csv'] = args.test_csv
        c.dataset['test_data_root_path'] = args.test_root_dir

        c.test_config['batch_size'] = args.batch_size
        c.test_config['num_workers'] = args.num_workers
        max_seq_len = c.dataset['max_seq_len'] 

        c.train_config['seed'] = 0

        testdataloader = test_dataloader(c, ap, max_seq_len=max_seq_len)

    
        # load model
        model = return_model(c)
        enablePrint()
        if checkpoint_path is not None:
            print("Loading checkpoint: %s" % checkpoint_path)
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'])
                print("Model Sucessful Load !")
            except Exception as e:
                raise ValueError("You need pass a valid checkpoint, may be you need check your config.json because de the of this checkpoint cause the error: "+ e) 
        blockPrint()
        # convert model from cuda
        if cuda:
            model = model.cuda()

        model.train(False)

        vote, targets, wav_path = test(criterion, ap, model, c, testdataloader, cuda=cuda, confusion_matrix=True, debug=debug, simples_vote=args.simples_vote)
        # print(vote)
        wav_files.append(wav_path)
        votes.append(vote)
        if len(wav_files):
            if wav_files[-1] != wav_files[0]:
                raise ValueError("Diferents files  or order for the test in diferrents seeds or folds")
    
    # mean vote, and round is necessary if use composite vote
    preds = np.mean(np.array(votes), axis=0)
    # print(preds)
    if not return_potential:
        preds = preds.round()

    file_names = wav_files[0]

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
        return preds, file_names
    else: 
        df = pd.DataFrame({'filename': file_names, 'label':preds.astype(int)})
        df['label'] = df['label'].replace(int(c.dataset['control_class']), 'negative', regex=True).replace(int(c.dataset['patient_class']), 'positive', regex=True)
        if args.output_csv:
            out_csv_path = args.output_csv
        else:
            out_csv_path = os.path.join(args.experiment_dir, os.path.basename(c.dataset['test_csv']))
        
        df.to_csv(out_csv_path, index=False)


if __name__ == '__main__':
    # python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1/spiraconv/checkpoint_1068.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1/spiraconv/config.json  --batch_size 5 --num_workers 2 --no_insert_noise True

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_csv', type=str, required=True,
                        help="test csv example: ../SPIRA_Dataset_V1/metadata_test.csv")
    parser.add_argument('-r', '--test_root_dir', type=str, required=True,
                        help="Test root dir example: ../SPIRA_Dataset_V1/")
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
    parser.add_argument('--output_csv', type=str, default=None,
                        help="CSV output")
                        
    args = parser.parse_args()

    
    run_test_all_seeds(args, cuda=True, debug=args.debug, return_potential=False)