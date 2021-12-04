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

def train_model(data_, test_, y_, folds_, parms):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        clf = LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.08,
            num_leaves=2 ** 5,
            colsample_bytree=.65,
            subsample=.9,
            max_depth=5,
            #             max_bin=250,
            reg_alpha=.3,
            reg_lambda=.3,
            min_split_gain=.01,
            min_child_weight=2,
            silent=-1,
            verbose=-1,
        )

        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=40  # 30
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))

    test_['isDefault'] = sub_preds

    return oof_preds, test_[['loan_id', 'isDefault']], feature_importance_df


def objective(trial):
    ''' 使用optuna框架进行调参 '''
    X_train, X_test, y_train, y_test = train_test_split(train, y, train_size=0.3)  # 数据集划分
    param = {

        'metric': 'auc',
        'random_state': 48,
        'n_estimators': 4000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20, 50]),
        'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth': trial.suggest_int('cat_smooth', 1, 100)
    }

    lgb = LGBMClassifier(**param)
    lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    pred_lgb = lgb.predict(X_test)
    auc = roc_auc_score(y_test, pred_lgb)
    return auc


# if __name__ == '__main__' :
file_processor = LoadSave(dir_name='./data/')

train = file_processor.load_data(
        file_name='total_train.pkl')
test = file_processor.load_data(
        file_name='total_test.pkl')
print(train.shape)
print(test.shape)

y = train['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=546789)

study=optuna.create_study(direction='minimize')
n_trials=50 # try50次
study.optimize(objective, n_trials=n_trials)
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_parallel_coordinate(study)#
optuna.visualization.plot_param_importances(study)#

params=study.best_params
params['metric'] = 'auc'

oof_preds, test_preds, importances = train_model(train, test, y, folds, params)

test_preds.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('nn2.csv', index=False)



