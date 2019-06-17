import pandas as pd
import numpy as np


def analysis_data(path):

    print('-------analysis train and test--------')
    train = pd.read_csv(path + "data/train.csv")
    test = pd.read_csv(path + "data/test.csv")

    print(train.head())
    print(test.head())

    a = train.groupby(['listing_id','repay_date'],as_index=False)['due_amt'].agg({'repay_amt':'sum'})
    print(a.shape)
    print(a.head())

    print("train columns: \n", list(train.columns))
    print("test columns: \n", list(test.columns))
    print("train shape: ", train.shape)
    print("test shape: ", test.shape)
    print("train summary: \n", train.describe())
    print("test summary: \n", test.describe())

    print("train user_id nunique: ", train['user_id'].nunique())
    print("test user_id nunique: ", test['user_id'].nunique())

    print("train listing_id nunique: ", train['listing_id'].nunique())
    print("test listing_id nunique: ", test['listing_id'].nunique())

    print("train auditing_date: ", train['auditing_date'].min(),train['auditing_date'].max())
    print("test auditing_date: ", test['auditing_date'].min(), test['auditing_date'].max())

    print("train due_date: ", train['due_date'].min(), train['due_date'].max())
    print("test due_date: ", test['due_date'].min(), test['due_date'].max())

    print(len(set(train.user_id) & set(test.user_id)))

    train['label'] = -1
    train['due_date'] = pd.to_datetime(train['due_date'])
    ind = train[train['repay_date'] != '\\N'].index
    train.ix[ind, 'label'] = (train.ix[ind, 'due_date'] - pd.to_datetime(train.ix[ind, 'repay_date'])).map(lambda x: x.days)

    train['mm'] = 0
    train.ix[ind, 'mm'] = train.ix[ind, 'due_amt'].astype(float) - train.ix[ind, 'repay_amt'].astype(float)

    a = train.groupby(['label'],as_index= False)['user_id'].agg({'count':'count'})
    a['r'] = a['count'] / len(train)
    print(a)

    a = train.groupby(['mm'], as_index=False)['user_id'].agg({'count': 'count'})
    print(a)

def get_label(due_date, repay_date):
    if repay_date == '\\N':
        label = -1
    else:
        label = due_date - pd.to_datetime(repay_date)
    print(label)
    return label

