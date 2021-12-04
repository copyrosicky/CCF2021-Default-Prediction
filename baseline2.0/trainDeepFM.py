"""
本模块包含DeepFM的训练和评估函数
"""

import os
import time

from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from DeepFM import DeepFM
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from utils import LoadSave


def evaluate_model(model, valid_loader):
    '''
    :param model:
    :return: 使用auc评估模型精度
    '''
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for step, x in tqdm(enumerate(valid_loader)):
            cat_fea, num_fea, label = x[0], x[1], x[2]
            if torch.cuda.is_available():
                cat_fea, num_fea, label = cat_fea.cuda(), num_fea.cuda(), label.cuda()
            # 输出模型预测值
            logits = model(cat_fea, num_fea)
            logits = logits.view(-1).data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        return cur_auc


def train_model(model, train_loader, train_batch_size, learning_rate, weight_decay,  Epochs):
    '''
    :param model:
    :param train_loader
    :param learning_rate: 学习率
    :param weight_decay:  权重衰减
    :param train_batch_size:
    :param Epochs:  迭代轮数
    :return:
    '''
    # 指定多gpu运行
    if torch.cuda.is_available():
        model.cuda()


    loss_function = nn.BCELoss()
    # 默认使用Adam进行优化
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_batch_size, gamma=0.8)  # 每过step_size个epoch，做一次更新

    best_auc = 0.0
    for epoch in range(Epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        for step, x in enumerate(train_loader):
            cat_fea, num_fea, label = x[0], x[1], x[2]

            if torch.cuda.is_available():
                cat_fea, num_fea, label = cat_fea.cuda(), num_fea.cuda(), label.cuda()

            pred = model(cat_fea, num_fea)
            # print(pred.size())  # torch.Size([2, 1])
            pred = pred.view(-1)
            loss = loss_function(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    epoch+1, step+1, len(train_loader), train_loss_sum/(step+1), time.time() - start_time))

        scheduler.step()
        cur_auc = evaluate_model(model)
        if cur_auc > best_auc:
            best_auc = cur_auc
            os.makedirs('./save_model', exist_ok=True)
            torch.save(model.state_dict(), './save_model/deepfm.bin')


if __name__ == '__main__':
    file_processor = LoadSave(dir_name='./data/')

    data = file_processor.load_data(
        file_name='total_train.pkl')
    total_test = file_processor.load_data(
        file_name='total_test.pkl')
    print(data.shape)
    print(total_test.shape)

    # data = pd.concat([train, test], axis=0)
    # print(data.shape)


    # 数据预处理
    nan_features = ['known_outstanding_loan', 'known_dero', 'app_type']
    sparse_features = [ 'class', 'employer_type', 'industry', 'house_exist', 'censor_status',
                       'region', 'initial_list_status', 'policy_code']
    dense_features = [f for f in data.columns.tolist() if f not in ['loan_id', 'user_id', 'isDefault'] and sparse_features
                      and nan_features]
    # sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]

    # 处理缺失值

    # known_outstanding_loan    known_dero
    data.isnull().sum()
    # data = data.drop(['known_outstanding_loan', 'known_dero', 'app_type'], axis = 1 )
    data[sparse_features] = data[sparse_features].fillna('0',)
    data[dense_features] = data[dense_features].fillna(data[dense_features].median(),)
    target = ['isDefault']

    # 将类别数据转为数字
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 将连续值归一化
    for feat in tqdm(dense_features):
        mean = data[feat].mean()
        std = data[feat].std()
        data[feat] = (data[feat] - mean) / (std + 1e-12)
    # print(data.shape)
    # print(data.head())

    train, valid = train_test_split(data, test_size=0.1, random_state=42)
    print(train.shape)   # (540000, 40)
    print(valid.shape)   # (60000, 40)
    train_dataset = TensorDataset(torch.LongTensor(train[sparse_features].values),
                                  torch.FloatTensor(train[dense_features].values),
                                  torch.FloatTensor(train['isDefault'].values))

    train_batch_size = 10
    eval_batch_size = 10
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.LongTensor(valid[sparse_features].values),
                                  torch.FloatTensor(valid[dense_features].values),
                                  torch.FloatTensor(valid['isDefault'].values))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=eval_batch_size, shuffle=False)

    cat_fea_unique = [data[f].nunique() for f in sparse_features]

    model = DeepFM(cat_fea_unique, num_fea_size=len(dense_features))

    train_model(model, train_loader, train_batch_size = 10 , learning_rate = 0.01, weight_decay = 1e-4, Epochs = 10)
    a = 1
    cur_auc = evaluate_model(model, valid_loader)
