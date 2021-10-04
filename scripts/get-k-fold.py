
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd

def return_num_neg_pos(df):
  positive_df = df.loc[df['label'] == patient_class]
  negative_df = df.loc[df['label'] == control_class]
  print("Total:", len(df), "Negative Samples:", len(negative_df),"Positive  Samples:", len(positive_df))

def get_df(d):
  return pd.read_csv(d, sep=',')#.replace({'negative': control_class}, regex=True).replace({'positive': patient_class}, regex=True)

# Speech
fold_dir = "../Speech/dist/lab/5-fold/"
dataset_csvs =["../Speech/dist/lab/train.csv", "../Speech/dist/lab/devel.csv"] 

#Cough
# fold_dir = "../Tosse/dist/lab/5-fold/"
# dataset_csvs =["../Tosse/dist/lab/train.csv", "../Tosse/dist/lab/devel.csv"] 

control_class = 'negative' # 0
patient_class = 'positive' # 1
num_folds = 5
seed = 42

dataset_df = None
for d in dataset_csvs:
  if dataset_df is None:
    dataset_df = get_df(d)
  else:
    dataset_df = dataset_df.append(get_df(d), ignore_index=True)

dataset_df = dataset_df.reset_index(drop=True)
print("Dataset Lenght:", len(dataset_df))

positive_df = dataset_df.loc[dataset_df['label'] == patient_class]
negative_df = dataset_df.loc[dataset_df['label'] == control_class]
print("Negative Samples:", len(negative_df),"Positive  Samples:", len(positive_df))

# suffle positive and negative instances
positive_df = positive_df.sample(frac=1, random_state=seed).reset_index(drop=True)
negative_df = negative_df.sample(frac=1, random_state=seed).reset_index(drop=True)

start_positive = 0
step_positive = int(len(positive_df)/num_folds)

start_negative = 0
step_negative = int(len(negative_df)/num_folds)

count = 1
for i in range(1, num_folds+1):
  print("="*10, "Fold:", i, "="*10)
  if i == num_folds:
    fold_negative_sample = negative_df[start_negative:]
    fold_positive_sample = positive_df[start_positive:]
  else:
    # random sample for equal negative and positive instances
    end_neg = start_negative+step_negative
    fold_negative_sample = negative_df[start_negative:end_neg]
    start_negative = start_negative + step_negative
    

    end_pos = start_positive+step_positive
    fold_positive_sample = positive_df[start_positive:end_pos]
    start_positive = start_positive + step_positive

  # concate positive and sampled instances
  validation_set = pd.concat([fold_positive_sample, fold_negative_sample])
  # all not in validation set put in train set:
  train_set = dataset_df[~dataset_df['filename'].isin(validation_set['filename'])]

  # suffle valdiation and train df
  validation_set = validation_set.sample(frac=1, random_state=seed).reset_index(drop=True)
  train_set = train_set.sample(frac=1, random_state=seed).reset_index(drop=True)
  
  # print logs
  print("Train:")
  return_num_neg_pos(train_set)
  print("Validation:")
  return_num_neg_pos(validation_set)
  print("VALID+TRAIN Lenght:", len(validation_set)+len(train_set))

  # integrite checks
  df_check = pd.concat([train_set, validation_set]).reset_index(drop=True)
  
  # if have one valid sample in train
  if validation_set['filename'].isin(train_set['filename']).any():
    print("Validation sample in Trainset")
    exit()
  if df_check.duplicated().any():
    print("Error duplicated instance on train and validation")
    exit()
  if len(df_check) != len(dataset_df):
    print("Dataset lenght Problem")
    exit()

  os.makedirs(os.path.join(fold_dir, str(count)), exist_ok=True)
  train_set.to_csv(os.path.join(fold_dir, str(count), 'train.csv'), index=False)
  validation_set.to_csv(os.path.join(fold_dir, str(count), 'devel.csv'), index=False)
  count+=1
