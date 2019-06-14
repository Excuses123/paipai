from data_analysis import *


'''
作为回归任务:
预测目标(label):还款当天距离最后还款日期的时间间隔(天)
'''
def gen_train_data(args):

    path = args['path']

    # 1.训练数据、测试数据
    train = pd.read_csv(path + "data/train.csv")
    test = pd.read_csv(path + "data/test.csv")

    train['label'] = -1
    train['due_date'] = pd.to_datetime(train['due_date'])
    ind = train[train['repay_date'] != '\\N'].index
    train.ix[ind, 'label'] = (train.ix[ind, 'due_date'] - pd.to_datetime(train.ix[ind, 'repay_date'])).map(lambda x: x.days)

    test['due_date'] = pd.to_datetime(test['due_date'])


    return train.drop(columns = ['repay_date', 'repay_amt']), test

'''
user_id,listing_id,auditing_date,due_date,due_amt,repay_date,repay_amt
748147,3163926,2018-04-25,2018-05-25,72.1167,2018-05-25,72.1167
672952,3698760,2018-06-09,2018-07-09,258.7045,2018-07-08,258.7045
404196,2355665,2018-02-18,2018-03-18,307.927,
'''


