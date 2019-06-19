from data_analysis import *
from models import *
import os, gc, datetime
from sklearn.preprocessing import LabelEncoder


'''
作为回归任务:
预测目标(label):还款当天距离最后还款日期的时间间隔(天)
'''
lb = LabelEncoder()

def gen_data(args):

    path = args['path']
    # 1.训练数据、测试数据
    if os.path.exists(path + 'cache/train.csv') and os.path.exists(path + 'cache/test.csv'):
        train = pd.read_csv(path + 'cache/train.csv')
        test = pd.read_csv(path + 'cache/test.csv')
    else:
        if os.path.exists(path + 'cache/train_cache.csv') and os.path.exists(path + 'cache/test_cache.csv'):
            train_cache = pd.read_csv(path + 'cache/train_cache.csv')
            test_cache = pd.read_csv(path + 'cache/test_cache.csv')
        else:
            train_cache, test_cache = gen_train_data(path)

        train, test = add_features(train_cache, test_cache, path)
        train.to_csv(path + 'cache/train.csv', index=None)
        test.to_csv(path + 'cache/test.csv', index=None)

    return train, test

def gen_train_data(path):

    # 1.训练数据、测试数据
    train = pd.read_csv(path + "data/train.csv")
    test = pd.read_csv(path + "data/test.csv")

    train['label'] = -10
    ind = train[train['repay_date'] != '\\N'].index
    train.ix[ind, 'label'] = (pd.to_datetime(train.ix[ind, 'due_date']) - pd.to_datetime(train.ix[ind, 'repay_date'])).map(lambda x: x.days)

    test['due_date'] = pd.to_datetime(test['due_date'])

    if not os.path.exists(path + 'cache/'):
        os.mkdir(path + 'cache/')
    train.drop(columns=['repay_date', 'repay_amt']).to_csv(path + 'cache/train_cache.csv', index=None)
    test.to_csv(path + 'cache/test_cache.csv', index=None)

    gc.collect()

    return train.drop(columns = ['repay_date', 'repay_amt']), test


def add_features(train, test, path):

    train, test = add_listing_info_features(train, test, path)
    train, test = add_user_info_features(train, test, path)
    # train, test = add_user_taglist_features(train, test, path)
    train, test = add_user_behavior_features(train, test, path)
    train, test = add_user_repay_features(train, test, path)

    for data in [train, test]:
        data['auditing_date'] = pd.to_datetime(data['auditing_date'])
        data['insertdate'] = pd.to_datetime(data['insertdate'])
        data['auditing_date_insertdate'] = (data['auditing_date'] - data['insertdate']).map(lambda x: x.days)

    return train, test


def add_listing_info_features(train, test, path):
    print("==========add_listing_info_features==========")
    # 2.标的属性表（listing_info.csv）
    listing_info = pd.read_csv(path + "data/listing_info.csv")
    listing_info['principal/term'] = listing_info['principal'] / listing_info['term']

    train = pd.merge(train, listing_info[['listing_id', 'term', 'rate', 'principal']],
                     on=['listing_id'], how='left')
    test = pd.merge(test, listing_info[['listing_id', 'term', 'rate', 'principal']],
                     on=['listing_id'], how='left')

    a1 = listing_info.groupby(['user_id'], as_index=False)['term'].agg(
        {'user_listing_count': 'count', 'user_term_mean': 'mean',
         'user_term_min': 'min', 'user_term_max': 'max', 'user_term_sum': 'sum'})

    a2 = listing_info.groupby(['user_id'], as_index=False)['rate'].agg(
        {'user_rate_mean': 'mean','user_rate_min': 'min', 'user_rate_max': 'max', 'user_rate_var': 'var'})

    a3 = listing_info.groupby(['user_id'], as_index=False)['principal'].agg(
        {'user_principal_mean': 'mean', 'user_principal_min': 'min', 'user_principal_max': 'max',
         'user_principal_var': 'var', 'user_principal_sum': 'sum'})

    a4 = listing_info.groupby(['user_id'], as_index=False)['principal/term'].agg(
        {'user_principal/term_mean': 'mean', 'user_principal/term_min': 'min', 'user_principal/term_max': 'max',
         'user_principal/term_var': 'var', 'user_principal/term_sum': 'sum'})

    for a in [a1, a2, a3, a4]:
        train = pd.merge(train, a, on='user_id', how='left')
        test = pd.merge(test, a, on='user_id', how='left')

    return train, test

def add_user_info_features(train, test, path):
    print("==========add_user_info_features==========")
    # 3. 借款用户基础信息表（user_info.csv）
    user_info = pd.read_csv(path + "data/user_info.csv")

    user_info['reg_mon'] = lb.fit_transform(user_info['reg_mon'])
    user_info['gender'] = lb.fit_transform(user_info['gender'])
    user_info['cell_province'] = lb.fit_transform(user_info['cell_province'])
    user_info['id_province'] = lb.fit_transform(user_info['id_province'])
    user_info['id_city'] = lb.fit_transform(user_info['id_city'])

    user_info.sort_values(by=['user_id', 'insertdate'], inplace=True)
    a1 = user_info.ix[user_info.groupby(['user_id'])['insertdate'].tail(1).index, :]
    a2 = user_info.groupby(['user_id'], as_index=False)['gender'].agg({'user_info_count':'count'})

    for a in [a1,a2]:
        train = pd.merge(train, a, on=['user_id'], how='left')
        test = pd.merge(test, a, on=['user_id'], how='left')

    return train, test

def add_user_taglist_features(train, test, path):
    print("==========add_user_taglist_features==========")
    # 4. 用户画像标签列表（user_taglist.csv）
    user_taglist = pd.read_csv(path + "data/user_taglist.csv")

    a1 = user_taglist.groupby(['user_id'], as_index=False)['insertdate'].agg({'user_insertdate_count': 'count'})

    user_taglist.sort_values(by=['user_id', 'insertdate'], inplace=True)
    user_taglist = user_taglist.ix[user_taglist.groupby(['user_id'])['insertdate'].tail(1).index, :]

    user_taglist['taglist'] = user_taglist['taglist'].map(lambda x: x.split('|'))
    user_taglist['taglist_len'] = user_taglist['taglist'].map(lambda x: len(x))

    if os.path.exists(path + 'cache/user_taglist_word2vec.model'):
        tag_model = Word2Vec.load(path + 'cache/user_taglist_word2vec.model')
    else:
        tag_model = word2vec_fit(user_taglist['taglist'], user_taglist['taglist_len'].max())
        tag_model.save(path + 'cache/user_taglist_word2vec.model')


    return train, test

def add_user_behavior_features(train, test, path):
    print("==========add_user_behavior_features==========")
    # 5.借款用户操作行为日志表（user_behavior_logs.csv）
    user_behavior_logs = pd.read_csv(path + "data/user_behavior_logs.csv")

    a1 = user_behavior_logs.groupby(['user_id'], as_index=False)['behavior_type'].agg({'user_behavior_count':'count'})
    a2 = user_behavior_logs.groupby(['user_id', 'behavior_type']).size().unstack().reset_index()
    a2.columns = ['user_id'] + ['behavior_type_' + str(i) for i in a2.columns[1:]]

    for a in [a1, a2]:
        train = pd.merge(train, a, on = ['user_id'], how='left')
        test = pd.merge(test, a, on=['user_id'], how='left')

    return train, test

def add_user_repay_features(train, test, path):
    print("==========add_user_repay_features==========")
    # 6.用户还款日志表（user_repay_logs.csv）
    user_repay_logs = pd.read_csv(path + "data/user_repay_logs.csv")
    user_repay_logs['overdue'] = user_repay_logs['repay_date'].map(lambda x: 1 if x == '2200-01-01' else 0)

    user_repay_logs['gap_date'] = user_repay_logs[['due_date', 'repay_date']].apply(gen_repay_gap_day, axis = 1)

    a1 = user_repay_logs.groupby(['user_id'], as_index=False)['listing_id'].agg({
        'user_repay_count':'count','user_repay_nunique':'nunique'})
    a2 = user_repay_logs.groupby(['user_id'], as_index=False)['order_id'].agg(
        {'user_repay_order_id_mean': 'mean'})
    a3 = user_repay_logs.groupby(['user_id'], as_index=False)['due_amt'].agg(
        {'user_repay_due_amt_mean': 'mean', 'user_repay_due_amt_sum': 'sum', 'user_repay_due_amt_max': 'max'})
    a4 = user_repay_logs.groupby(['user_id', 'overdue']).size().unstack().reset_index().fillna(0)
    a4.columns = ['user_id'] + ['user_id_overdue_' + str(i) for i in a4.columns[1:]]
    a4['user_id_overdue_rate'] = a4['user_id_overdue_1'] / a4['user_id_overdue_0']

    a5 = user_repay_logs.groupby(['user_id'], as_index=False)['gap_date'].agg(
        {'user_repay_gap_date_mean': 'mean', 'user_repay_gap_date_sum': 'sum', 'user_repay_gap_date_max': 'max', 'user_repay_gap_date_var': 'var'})

    b1 = user_repay_logs.groupby(['listing_id'], as_index=False)['user_id'].agg({
        'listing_id_repay_count': 'count', 'listing_id_repay_nunique': 'nunique'})
    b2 = user_repay_logs.groupby(['listing_id'], as_index=False)['order_id'].agg(
        {'listing_id_repay_order_id_mean': 'mean'})
    b3 = user_repay_logs.groupby(['listing_id'], as_index=False)['due_amt'].agg(
        {'listing_id_repay_due_amt_mean': 'mean', 'listing_id_repay_due_amt_sum': 'sum', 'listing_id_repay_due_amt_max': 'max'})
    b4 = user_repay_logs.groupby(['listing_id', 'overdue']).size().unstack().reset_index().fillna(0)
    b4.columns = ['listing_id'] + ['listing_id_overdue_' + str(i) for i in b4.columns[1:]]
    b4['listing_id_overdue_rate'] = b4['listing_id_overdue_1'] / b4['listing_id_overdue_0']

    b5 = user_repay_logs.groupby(['listing_id'], as_index=False)['gap_date'].agg(
        {'listing_id_repay_gap_date_mean': 'mean', 'listing_id_repay_gap_date_sum': 'sum',
         'listing_id_repay_gap_date_max': 'max', 'listing_id_repay_gap_date_var': 'var'})

    for a in [a1, a2, a3, a4, a5]:
        train = pd.merge(train, a, on=['user_id'], how='left')
        test = pd.merge(test, a, on=['user_id'], how='left')

    for b in [b1, b2, b3, b4, b5]:
        train = pd.merge(train, b, on=['listing_id'], how='left')
        test = pd.merge(test, b, on=['listing_id'], how='left')

    return train, test

#####help function####
def gen_result(model, test, cols):
    print('===========gen result==========')
    test['pred'] = model.predict(test[cols]).astype(int)
    test['repay_date'] = test[['auditing_date', 'due_date', 'pred']].apply(gen_repay_date, axis=1)
    test['repay_amt'] = test[['due_amt', 'pred']].apply(gen_repay_amt, axis=1)

    return test[['listing_id', 'repay_amt', 'repay_date']]

def gen_repay_date(x):
    date = pd.to_datetime(x[1]) - datetime.timedelta(days=x[2])
    if x[2] < 0:
        return ''
    elif date <= pd.to_datetime(x[0]):
        return x[0]
    else:
        return str(date)[:10]

def gen_repay_amt(x):
    if x[1] < 0:
        return ''
    else:
        return str(x[0])

def gen_repay_gap_day(x):
    if x[1] == '2200-01-01':
        return np.nan
    else:
        return (pd.to_datetime(x[0]) - pd.to_datetime(x[1])).days
