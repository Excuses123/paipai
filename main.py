from argparse import ArgumentParser
from feature_extraction import *
import models


def parse_command_params():
    """
    命令行参数解析器
    :return:
    """
    ap = ArgumentParser()  # 创建解析器
    ap.add_argument('-s', '--show', default='no', help="if you want to visualize training process")
    ap.add_argument('-m', '--method', default='all', help='fit all or fit generator')
    ap.add_argument('-b', '--batch', default=64, help='batch size of training')
    ap.add_argument('-e', '--epochs', default=100, help='how many epochs to train')
    ap.add_argument('-l', '--loss', default='ce', help='which loss function to train')
    ap.add_argument('-p', '--path', default='../data/', help='data path')
    args_ = vars(ap.parse_args())
    return args_


def save_zip(result, path):
    result.to_csv(path)




if __name__ == "__main__":
    '''提供了2018年1月1日至2018年12月31日的标的第一期的还款数据作为训练集，
    需要选手预测2019年2月1日至2019年3月31日成交标的第一期的还款情况'''

    #时间序列，回归，或者分类(排除逾期的用户，剩下用户直接统计还款日作为结果) 二分类或多分类（逾期，逾期前1天、2天、3天还款）
    args = parse_command_params()

    analysis_data(args['path'])

    # train, test = gen_train_data(args['path'])
    # model = models.fit(train, args)
    # result = model.predict(test)
    # save_zip(result, args['path'])

    print('done !')


