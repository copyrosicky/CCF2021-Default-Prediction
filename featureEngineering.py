# -- coding: utf-8 --
# encoding: utf-8

import gc
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from utils import LoadSave

# 读入原始的训练与测试数据
# -------------------------
file_processor = LoadSave(dir_name='./data/')

test_public = file_processor.load_data(
            file_name='test_public.pkl')
train_data = file_processor.load_data(
            file_name='train_data.pkl')
train_inte = file_processor.load_data(
            file_name='train_inte.pkl')


# 编码处理 =================================================================================

cat_cols = ['employer_type', 'industry']
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    train_data[col] = lbl.transform(train_data[col])
    test_public[col] = lbl.transform(test_public[col])
    train_inte[col] = lbl.transform(train_inte[col])

col_to_drop = ['issue_date', 'earlies_credit_mon']
train_data = train_data.drop(col_to_drop, axis=1)
test_public = test_public.drop(col_to_drop, axis=1)
train_inte = train_inte.drop(col_to_drop, axis=1)

# sub class feature =============================================================================
# 在计算伪标签之前，将main data中的class进行聚类，保证维度和train_inte中的subclass一致
# 构造subclass强特征 之前直接忽略掉了 在复盘的时候才发现 ORZ
print(train_inte['sub_class'].unique())
print(train_data['class'].unique())
print(test_public['class'].unique())
'''
['A1' 'A2' 'A3' 'A4' 'A5' 'B1' 'B2' 'B3' 'B4' 'B5' 'C1' 'C2' 'C3' 'C4' 'C5' 'D1' 'D2' 'D3' 'D4' 'D5' 'E1' 'E2' 
'E3' 'E4' 'E5' 'F1' 'F2' 'F3' 'F4' 'F5' 'G1' 'G2' 'G3' 'G4' 'G5']
共七大类其中每大类中有五小类，采用聚类方案
'''


def feature_Kmeans(data, label):
    mms = MinMaxScaler()
    feats = [f for f in data.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    data = data[feats]
    mmsModel = mms.fit_transform(data.loc[data['class'] == label])
    clf = KMeans(5, random_state=2021)
    pre = clf.fit(mmsModel)
    test = pre.labels_
    final_data = pd.Series(test, index=data.loc[data['class'] == label].index)
    if label == 1:
        final_data = final_data.map({0: 'A1', 1: 'A2', 2: 'A3', 3: 'A4', 4: 'A5'})
    elif label == 2:
        final_data = final_data.map({0: 'B1', 1: 'B2', 2: 'B3', 3: 'B4', 4: 'B5'})
    elif label == 3:
        final_data = final_data.map({0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5'})
    elif label == 4:
        final_data = final_data.map({0: 'D1', 1: 'D2', 2: 'D3', 3: 'D4', 4: 'D5'})
    elif label == 5:
        final_data = final_data.map({0: 'E1', 1: 'E2', 2: 'E3', 3: 'E4', 4: 'E5'})
    elif label == 6:
        final_data = final_data.map({0: 'F1', 1: 'F2', 2: 'F3', 3: 'F4', 4: 'F5'})
    elif label == 7:
        final_data = final_data.map({0: 'G1', 1: 'G2', 2: 'G3', 3: 'G4', 4: 'G5'})
    return final_data


# 训练集合并
train_data1 = feature_Kmeans(train_data, 1)
train_data2 = feature_Kmeans(train_data, 2)
train_data3 = feature_Kmeans(train_data, 3)
train_data4 = feature_Kmeans(train_data, 4)
train_data5 = feature_Kmeans(train_data, 5)
train_data6 = feature_Kmeans(train_data, 6)
train_data7 = feature_Kmeans(train_data, 7)
train_dataall = pd.concat(
    [train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7]).reset_index(drop=True)
train_data['sub_class'] = train_dataall
# 测试集合并
test_data1 = feature_Kmeans(test_public, 1)
test_data2 = feature_Kmeans(test_public, 2)
test_data3 = feature_Kmeans(test_public, 3)
test_data4 = feature_Kmeans(test_public, 4)
test_data5 = feature_Kmeans(test_public, 5)
test_data6 = feature_Kmeans(test_public, 6)
test_data7 = feature_Kmeans(test_public, 7)
test_dataall = pd.concat(
    [test_data1, test_data2, test_data3, test_data4, test_data5, test_data6, test_data7]).reset_index(drop=True)
test_public['sub_class'] = test_dataall

cat_cols = ['sub_class']
for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    train_data[col] = lbl.transform(train_data[col])
    test_public[col] = lbl.transform(test_public[col])
    train_inte[col] = lbl.transform(train_inte[col])


# 特征构造====================================================================================

train_data['post_code_to_mean_interst'] = train_data.groupby(['post_code'])['interest'].transform('mean')
test_public['post_code_to_mean_interst'] = test_public.groupby(['post_code'])['interest'].transform('mean')
train_inte['post_code_to_mean_interst'] = train_inte.groupby(['post_code'])['interest'].transform('mean')

train_data['recircle_u_b_std'] = train_data.groupby(['recircle_u'])['recircle_b'].transform('std')
test_public['recircle_u_b_std'] = test_public.groupby(['recircle_u'])['recircle_b'].transform('std')
train_inte['recircle_u_b_std'] = train_inte.groupby(['recircle_u'])['recircle_b'].transform('std')

train_data['early_return_amount_early_return'] = train_data['early_return_amount'] / train_data['early_return']
test_public['early_return_amount_early_return'] = test_public['early_return_amount'] / test_public['early_return']
train_inte['early_return_amount_early_return'] = train_inte['early_return_amount'] / train_inte['early_return']

# 可能出现极大值和空值
train_data['early_return_amount_early_return'][np.isinf(train_data['early_return_amount_early_return'])] = 0
test_public['early_return_amount_early_return'][np.isinf(test_public['early_return_amount_early_return'])] = 0
train_inte['early_return_amount_early_return'][np.isinf(train_inte['early_return_amount_early_return'])] = 0

# 还款利息
train_data['total_loan_monthly_payment'] = train_data['monthly_payment'] * train_data['year_of_loan'] * 12 - train_data[
    'total_loan']
test_public['total_loan_monthly_payment'] = test_public['monthly_payment'] * test_public['year_of_loan'] * 12 - \
                                            test_public['total_loan']
train_inte['total_loan_monthly_payment'] = train_inte['monthly_payment'] * train_inte['year_of_loan'] * 12 - train_inte[
    'total_loan']

# 取train_data和train_inte特征交集，train_inte将不包含的特征使用nan值填补
tr_cols = set(train_data.columns)
same_col = list(tr_cols.intersection(set(train_inte.columns)))
train_inteSame = train_inte[same_col].copy()
Inte_add_cos = list(tr_cols.difference(set(same_col)))
for col in Inte_add_cos:
    train_inteSame[col] = np.nan



# 伪标签学习      以train_data特征训练模型->预测train_inteSame =============================================


def train_model(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    feature_importance_df = pd.DataFrame()
    # feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault','policy_code','del_in_18month'] ]
    feats = [f for f in data_.columns if f not in ['loan_id', 'user_id', 'isDefault']]
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        cat_feats = {'industry', 'employer_type'}
        clf = LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.08,  # 0.07
            num_leaves=2 ** 5 + 1,
            colsample_bytree=.65,
            subsample=.9,
            max_depth=5,  # 5
            # max_bin=250,
            reg_alpha=.3,
            reg_lambda=.3,
            min_split_gain=.01,
            min_child_weight=2,
            silent=-1,
            verbose=-1,
        )
        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],  # categorical_feature=cat_feats,
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

import matplotlib.pyplot as plt
import seaborn as sns
def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

SEED = 7786
y = train_data['isDefault']
train_data.shape
folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds, IntePre, importances = train_model(train_data, train_inteSame, y, folds)


#roc_auc_score
from sklearn.metrics import roc_auc_score
IntePre['isDef'] = train_inte['is_default']
roc_auc_score(IntePre['isDef'],IntePre.isDefault)     # 0.6846

InteId = IntePre.loc[IntePre.isDefault<0.05, 'loan_id'].tolist()

## 选择阈值0.05，从internet表中提取预测小于该概率的样本，并对不同来源的样本赋予来源值
InteId = IntePre.loc[IntePre.isDefault<0.05, 'loan_id'].tolist()
#新增来源域分类特征
train_data['dataSourse'] = 1
test_public['dataSourse'] = 1
train_inteSame['dataSourse'] = 0
train_inteSame['isDefault'] = train_inte['is_default']
use_te = train_inteSame[train_inteSame.loan_id.isin( InteId )].copy()
#连接表
data = pd.concat([ train_data,test_public,use_te]).reset_index(drop=True)

for method in ['mean', 'std', 'sum', 'median']:
    for col in ['employer_type', 'industry', 'issue_date_month', 'issue_date_dayofweek', 'earliesCreditMon',
                'earliesCreditYear', 'region']:
        data[f'label_{method}_' + str(col)] = data.groupby(col)['isDefault'].transform(method)


# IntePre.isDefault
plt.figure(figsize=(16,6))
plt.title("Distribution of Default values IntePre")
sns.distplot(IntePre['isDefault'],color="black", kde=True,bins=120, label='train_data')
plt.legend();plt.show()

train = data[data['isDefault'].notna()]
test  = data[data['isDefault'].isna()]

display_importances(importances)




del data
del train_data,test_public

y = train['isDefault']
folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds, test_preds, importances = train_model(train, test, y, folds)
test_preds.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']]


# save data =========================================================
# =====================================================================
from utils import LoadSave
file_processor = LoadSave(dir_name='./data/')

file_processor.save_data(
            file_name='sub.pkl', data_file=test_preds)





