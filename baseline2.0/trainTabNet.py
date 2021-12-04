# -- coding: utf-8 --
# encoding: utf-8

import os
import time
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from lightgbm import LGBMClassifier
import optuna
import optuna.integration.lightgbm as oplgb

from DeepFM import DeepFM
from utils import LoadSave

from torch import optim
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_tabnet.pretraining import TabNetPretrainer

SEED = 6657
MAX_EPOCH = 120

file_processor = LoadSave(dir_name='./data/')

train = file_processor.load_data(
        file_name='total_train.pkl')
test = file_processor.load_data(
        file_name='total_test.pkl')
print(train.shape)
print(test.shape)

y_train = train['isDefault']

def fill_na(data):
    nan_features = ['known_outstanding_loan', 'known_dero', 'app_type']
    sparse_features = [ 'class', 'employer_type', 'industry', 'house_exist', 'censor_status',
                           'region', 'initial_list_status', 'policy_code']
    dense_features = [f for f in data.columns.tolist() if f not in ['loan_id', 'user_id', 'isDefault'] and sparse_features
                          and nan_features]
    data[sparse_features] = data[sparse_features].fillna('0', )
    data[dense_features] = data[dense_features].fillna(data[dense_features].median(), )
    return data

train = fill_na(train)
test = fill_na(test)

# folds = KFold(n_splits=5, shuffle=True, random_state=546789)


unsupervised_model = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax'  # "sparsemax"
    )
x_train = train[[f for f in train.columns if f not in ['isDefault','loan_id','user_id'] ]]
x_test = test[[f for f in train.columns if f not in ['isDefault','loan_id','user_id']]]
unsupervised_model.fit(
            X_train=  x_train,
            eval_set=[x_test],
            pretraining_ratio=0.8,
        )

test.isnull().sum()
clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, # how to use learning rate scheduler
                      "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax' # This will be overwritten if using pretrain model
)


clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    from_unsupervised=unsupervised_model
)

