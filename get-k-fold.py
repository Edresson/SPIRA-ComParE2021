
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd

def return_num_neg_pos(df):
  positive_df = df.loc[df['label'] == patient_class]
  negative_df = df.loc[df['label'] == control_class]
  print("Negative Samples:", len(negative_df),"Positive  Samples:", len(positive_df))

def get_df(d):
  return pd.read_csv(d, sep=',')#.replace({'negative': control_class}, regex=True).replace({'positive': patient_class}, regex=True)

# outdir
fold_dir = "../Speech/dist/lab/5-fold"

dataset_csvs =["../Speech/dist/lab/train.csv", "../Speech/dist/lab/devel.csv"] 

control_class = 'negative' # 0
patient_class = 'positive' # 1


df = None
for d in dataset_csvs:
  if df is None:
    df = get_df(d)
  else:
    df = df.append(get_df(d), ignore_index=True)


num_folds = 5


positive_df = df.loc[df['label'] == patient_class]
negative_df = df.loc[df['label'] == control_class]
print("Negative Samples:", len(negative_df),"Positive  Samples:", len(positive_df))


count = 1
for i in range(1, num_folds+1):
  # random sample for equal negative and positive instances
  negative_sample = negative_df.sample(n=len(positive_df), random_state=i)
  # get no sampled instances 
  not_in_neg_df = negative_df[~negative_df['filename'].isin(negative_sample['filename'])]

  print("No sampled Instances: ", len(not_in_neg_df))

  # concate positive and sampled instances
  df_aux = pd.concat([positive_df, negative_sample])
  # get balanced and random stratify train and test instances
  trainX, testX, _, _ = train_test_split(df_aux, df_aux["label"], random_state=i, test_size=200, stratify=df_aux["label"])
  # add no sampled instances on train dataset
  trainX = pd.concat([trainX, not_in_neg_df])

  if trainX.duplicated().any():
    print("Error duplicated instance on trainX")
    exit()

  print("TrainX:")
  return_num_neg_pos(trainX)
  print("TestX:")
  return_num_neg_pos(testX)
  
  os.makedirs(os.path.join(fold_dir, str(count)), exist_ok=True)
  trainX.to_csv(os.path.join(fold_dir, str(count), 'train.csv'), index=False)
  testX.to_csv(os.path.join(fold_dir, str(count), 'devel.csv'), index=False)
  count+=1