import os
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, recall_score

control_class =  0
patient_class =  1

def get_df(d):
  return pd.read_csv(d, sep=',').replace({'negative': control_class}, regex=True).replace({'positive': patient_class}, regex=True)


# target_csv = "../Speech/dist/lab/devel.csv"
# predicted_csv = "../Speech/Experiments_Final_kfolds/Experiment-1/devel.csv"
# predicted_csv = "../Speech/Experiments_Final/Experiment-3/devel.csv"

target_csv = "../Tosse/dist/lab/devel.csv"
predicted_csv = "../Tosse/Experiments_Final_kfolds/Experiment-2/devel.csv"
# predicted_csv = "../Tosse/Experiments_Final/Experiment-2/devel.csv"

target_df = get_df(target_csv).values.tolist()
predicted_df = get_df(predicted_csv).values.tolist()

targets = np.array(target_df)[:, 1].astype(int).tolist()
preds = np.array(predicted_df)[:, 1].astype(int).tolist()

f1 = f1_score(targets, preds)
uar = recall_score(targets, preds, average='macro')


print("F1:", f1, "UAR:", uar)


