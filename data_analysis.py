import pandas as pd
import numpy as np

def gen_data(path):

    # 1.训练数据、测试数据
    train = pd.read(path + "train.csv")
    test = pd.read(path + "test.csv")
    # 2.标的属性表（listing_info.csv）
    listing_info = pd.read(path + "listing_info.csv")
    # 3. 借款用户基础信息表（user_info.csv）
    user_info = pd.read(path + "user_info.csv")
    # 4. 用户画像标签列表（user_taglist.csv）
    user_taglist = pd.read(path + "user_taglist.csv")
    # 5.借款用户操作行为日志表（user_behavior_logs.csv）
    user_behavior_logs = pd.read(path + "user_behavior_logs.csv")
    # 6.用户还款日志表（user_repay_logs.csv）
    user_repay_logs = pd.read(path + "user_repay_logs.csv")

    return train

def analysis_data(path):

    print('-------analysis train and test--------')
    train = pd.read_csv(path + "data/train.csv")
    test = pd.read_csv(path + "data/test.csv")

    print(train.head())
    print(test.head())

    # print("train columns: \n", list(train.columns))
    # print("test columns: \n", list(test.columns))
    # print("train shape: ", train.shape)
    # print("test shape: ", test.shape)
    # print("train summary: \n", train.describe())
    # print("test summary: \n", test.describe())
    #
    # print("train user_id nunique: ", train['user_id'].nunique())
    # print("test user_id nunique: ", test['user_id'].nunique())
    #
    # print("train listing_id nunique: ", train['listing_id'].nunique())
    # print("test listing_id nunique: ", test['listing_id'].nunique())
    #
    # print("train auditing_date: ", train['auditing_date'].min(),train['auditing_date'].max())
    # print("test auditing_date: ", test['auditing_date'].min(), test['auditing_date'].max())
    #
    # print("train due_date: ", train['due_date'].min(), train['due_date'].max())
    # print("test due_date: ", test['due_date'].min(), test['due_date'].max())

    # print(len(set(train.user_id) & set(test.user_id)))

    print(len(train[train.repay_date == "\\N"])/len(train))
    train = train[train.repay_date != "\\N"]
    train['due_date'] = pd.to_datetime(train['due_date'])
    train['repay_date'] = pd.to_datetime(train['repay_date'])
    train['gap_day'] = train['due_date'] - train['repay_date']

    a = train.groupby(['gap_day'],as_index= False)['user_id'].agg({'count':'count'})
    a['r'] = a['count'] / len(train)
    print(a)


