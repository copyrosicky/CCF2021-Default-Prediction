# -- coding: utf-8 --
# encoding: utf-8
'''
本模块主要包括了数据预处理和lebel encoding的代码
'''

import matplotlib.pyplot as plt
import seaborn as sns
import gc
import re
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.model_selection import StratifiedKFold
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

from utils import LoadSave

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


# 时间格式处理 =================================================
# ===============================================================
def workYearDIc(x):
    if str(x) == 'nan':
        return -1
    x = x.replace('< 1', '0')
    return int(re.search('(\d+)', x).group())


def findDig(val):
    fd = re.search('(\d+-)', val)
    if fd is None:
        return '1-' + val
    return val + '-01'


class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}

def timeProcesser(data) :
    timeMax = pd.to_datetime('1-Dec-21')
    data['work_year'] = data['work_year'].map(workYearDIc)
    data['class'] = data['class'].map(class_dict)

    data['earlies_credit_mon'] = pd.to_datetime(data['earlies_credit_mon'].map(findDig))
    data.loc[train_data['earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] = data.loc[train_data[
                                                                                                      'earlies_credit_mon'] > timeMax, 'earlies_credit_mon'] + pd.offsets.DateOffset(
    years=-100)

    data['issue_date'] = pd.to_datetime(data['issue_date'])
    data['issue_date_month'] = data['issue_date'].dt.month
    data['issue_date_dayofweek'] = data['issue_date'].dt.dayofweek
    data['earliesCreditMon'] = data['earlies_credit_mon'].dt.month
    data['earliesCreditYear'] = data['earlies_credit_mon'].dt.year
    return data


train_data = timeProcesser(train_data)
test_public = timeProcesser(test_public)

# Internet数据处理
train_inte['work_year'] = train_inte['work_year'].map(workYearDIc)
train_inte['class'] = train_inte['class'].map(class_dict)
train_inte['earlies_credit_mon'] = pd.to_datetime(train_inte['earlies_credit_mon'])
train_inte['issue_date'] = pd.to_datetime(train_inte['issue_date'])

train_inte['issue_date_month'] = train_inte['issue_date'].dt.month
train_inte['issue_date_dayofweek'] = train_inte['issue_date'].dt.dayofweek
train_inte['earliesCreditMon'] = train_inte['earlies_credit_mon'].dt.month
train_inte['earliesCreditYear'] = train_inte['earlies_credit_mon'].dt.year


# target encoding  ====================================================
# =====================================================================
cat_cols = ['employer_type', 'industry']

for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    train_data[col] = lbl.transform(train_data[col])
    test_public[col] = lbl.transform(test_public[col])

    # Internet处理
    train_inte[col] = lbl.transform(train_inte[col])

for col in cat_cols :
    # internet data ????
    ce_target = ce.TargetEncoder(cols = [col])
    ce_target.fit(train_data[col], train_data['isDefault'])
    ce_target.transform(train_data[col], train_data['isDefault'])
    ce_target.transform(test_public[col], train_data['isDefault'])


train_data.info()

# impact encoding =================================================
# ==================================================================
def impact_coding(data, feature, target):
    '''
    In this implementation we get the values and the dictionary as two different steps.
    This is just because initially we were ignoring the dictionary as a result variable.

    In this implementation the KFolds use shuffling. If you want reproducibility the cv
    could be moved to a parameter.
    '''
    n_folds = 20
    n_inner_folds = 10
    impact_coded = pd.Series()

    oof_default_mean = data[target].mean()  # Gobal mean to use by default (you could further tune this)

    kf = KFold(n_splits=n_folds, shuffle=True)
    oof_mean_cv = pd.DataFrame()
    split = 0

    for infold, oof in kf.split(data[feature]):
        # impact_coded_cv = pd.Series()
        kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
        inner_split = 0
        inner_oof_mean_cv = pd.DataFrame()

        oof_default_inner_mean = data.iloc[infold][target].mean()

        for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
            # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
            oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
            # impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
            #     lambda x: oof_mean[x[feature]]
            #     if x[feature] in oof_mean.index
            #     else oof_default_inner_mean
            #     , axis=1))

            # Also populate mapping (this has all group -> mean for all inner CV folds)
            inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
            inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
            inner_split += 1

        # Also populate mapping
        oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
        oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
        split += 1

        impact_coded = impact_coded.append(data.iloc[oof].apply(
            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
            if x[feature] in inner_oof_mean_cv.index
            else oof_default_mean
            , axis=1))

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean


# Apply the encoding to training and test data, and preserve the mapping
impact_coding_map = {}
for f in cat_cols:
    print("Impact coding for {}".format(f))
    train_data["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(train_data, f,target = 'isDefault')
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_mean = impact_coding_map[f]
    # mapping : list
    # 对test data 进行 target encoding
    test_public["impact_encoded_{}".format(f)] = test_public.apply(lambda x: mapping[x[f]]
    if x[f] in mapping
    else default_mean , axis=1)

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

