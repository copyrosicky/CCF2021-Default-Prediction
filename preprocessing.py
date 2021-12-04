# -- coding: utf-8 --
# encoding: utf-8

'''
本模块主要包括了数据预处理和lebel encoding的代码
'''

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import gc
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# preprocessing ===================================================
# ==================================================================
# 读入原始的训练与测试数据

TRAIN_PATH = './data/'
TEST_PATH = './data/'

train_data = pd.read_csv(TRAIN_PATH + 'train_public.csv')
train_inte = pd.read_csv(TRAIN_PATH + 'train_internet.csv')
test_public = pd.read_csv(TEST_PATH + 'test_public.csv')

pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)


def workYearDIc(x):
    if str(x) == 'nan':
        return 0
    x = x.replace('< 1', '0')
    return int(re.search('(\d+)', x).group())


def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'


def findDig2(x):
    if x >= pd.to_datetime('2021-01-01'):
        t = '20' + str(x)[8:10] + '-' + str(x)[5:7] + '-01'
        # print('t=', t)
        return pd.to_datetime(t)
    return x


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns



# 缺失值补充
loss_numerical_feas = ['recircle_u', 'pub_dero_bankrup', 'debt_loan_ratio']
f_feas = ['f0', 'f1', 'f2', 'f3', 'f4']

train_data[loss_numerical_feas] = train_data[loss_numerical_feas].fillna(train_data[loss_numerical_feas].median())
train_data[f_feas] = train_data[f_feas].fillna(train_data[f_feas].median())
train_data['post_code'] = train_data['post_code'].fillna(train_data['post_code'].mode()[0])
train_data.loc[train_data['debt_loan_ratio'] <= 0, 'debt_loan_ratio'] = 0
for f in f_feas:
    train_data[f'industry_to_mean_{f}'] = train_data.groupby('industry')[f].transform('mean')

test_public[loss_numerical_feas] = test_public[loss_numerical_feas].fillna(test_public[loss_numerical_feas].median())
test_public['post_code'] = test_public['post_code'].fillna(test_public['post_code'].mode()[0])
test_public.loc[test_public['debt_loan_ratio'] <= 0, 'debt_loan_ratio'] = 0
test_public[f_feas] = test_public[f_feas].fillna(test_public[f_feas].median())
for f in f_feas:
    test_public[f'industry_to_mean_{f}'] = test_public.groupby('industry')[f].transform('mean')

train_inte[loss_numerical_feas] = train_inte[loss_numerical_feas].fillna(train_inte[loss_numerical_feas].median())
train_inte['post_code'] = train_inte['post_code'].fillna(train_inte['post_code'].mode()[0])
train_inte['title'] = train_inte['title'].fillna(train_inte['title'].mode()[0])
train_inte.loc[train_inte['debt_loan_ratio'] <= 0, 'debt_loan_ratio'] = 0
train_inte[f_feas] = train_inte[f_feas].fillna(train_inte[f_feas].median())
for f in f_feas:
    train_inte[f'industry_to_mean_{f}'] = train_inte.groupby('industry')[f].transform('mean')


class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}

timeMax = pd.to_datetime('1-Dec-21')
train_data['work_year'] = train_data['work_year'].map(workYearDIc)

test_public['work_year'] = test_public['work_year'].map(workYearDIc)

train_data['class'] = train_data['class'].map(class_dict)
test_public['class'] = test_public['class'].map(class_dict)

train_data['earlies_credit_mon'] = pd.to_datetime(train_data['earlies_credit_mon'].map(findDig))
test_public['earlies_credit_mon'] = pd.to_datetime(test_public['earlies_credit_mon'].map(findDig))

train_data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = train_data.loc[train_data[
                                                                                                      'earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(
    years=-100)
test_public.loc[test_public['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = test_public.loc[test_public[
                                                                                                         'earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(
    years=-100)
train_data['issue_date'] = pd.to_datetime(train_data['issue_date'])
test_public['issue_date'] = pd.to_datetime(test_public['issue_date'])

# Internet数据处理
train_inte['work_year'] = train_inte['work_year'].map(workYearDIc)
train_inte['class'] = train_inte['class'].map(class_dict)

train_inte['earlies_credit_mon'] = pd.to_datetime(train_inte['earlies_credit_mon'])
train_inte['issue_date'] = pd.to_datetime(train_inte['issue_date'])

train_data['issue_date_month'] = train_data['issue_date'].dt.month
train_data['issue_date_year'] = train_data['issue_date'].dt.year
test_public['issue_date_month'] = test_public['issue_date'].dt.month
test_public['issue_date_year'] = test_public['issue_date'].dt.year
train_data['issue_date_dayofweek'] = train_data['issue_date'].dt.dayofweek
test_public['issue_date_dayofweek'] = test_public['issue_date'].dt.dayofweek

train_data['earliesCreditMon'] = train_data['earlies_credit_mon'].dt.month
test_public['earliesCreditMon'] = test_public['earlies_credit_mon'].dt.month
train_data['earliesCreditYear'] = train_data['earlies_credit_mon'].dt.year
test_public['earliesCreditYear'] = test_public['earlies_credit_mon'].dt.year

train_inte['issue_date_month'] = train_inte['issue_date'].dt.month
train_inte['issue_date_dayofweek'] = train_inte['issue_date'].dt.dayofweek
train_inte['issue_date_year'] = train_inte['issue_date'].dt.year

train_inte['earliesCreditMon'] = train_inte['earlies_credit_mon'].dt.month
train_inte['earliesCreditYear'] = train_inte['earlies_credit_mon'].dt.year



# save data =========================================================
# =====================================================================
from utils import LoadSave
file_processor = LoadSave(dir_name='./data/')

file_processor.save_data(
            file_name='test_public.pkl', data_file=test_public)
file_processor.save_data(
            file_name='train_data.pkl', data_file=train_data)
file_processor.save_data(
            file_name='train_inte.pkl', data_file=train_inte)


