from argparse import ArgumentParser
from feature_extraction import *
import models
import zipfile, os, warnings

warnings.filterwarnings("ignore")

def parse_command_params():
    """
    命令行参数解析器
    :return:
    """
    ap = ArgumentParser()  # 创建解析器
    ap.add_argument('-p', '--path', default='../data/', help='data path')
    ap.add_argument('-o', '--output', default='result.csv', help='output result')
    ap.add_argument('-m', '--model', default='lgb', help='use model')
    ap.add_argument('-r', '--round', default=100, help='model train rounds')
    ap.add_argument('-lr', '--learning_rate', default=0.05, help='learning rate')
    args_ = vars(ap.parse_args())
    return args_

def save_zip(result, args):
    print('==========save zip result!============')
    path, filename = args['path'], args['output']
    if not os.path.exists(path + 'output/'):
        os.mkdir(path + 'output/')
    os.chdir(path + "output/")
    result.to_csv(filename, index = None)
    zip = zipfile.ZipFile(filename.split('.')[0] + '.zip', "w", zipfile.ZIP_DEFLATED)
    zip.write(filename)
    zip.close()



if __name__ == "__main__":
    '''提供了2018年1月1日至2018年12月31日的标的第一期的还款数据作为训练集，
    需要选手预测2019年2月1日至2019年3月31日成交标的第一期的还款情况'''

    #时间序列，回归，或者分类(排除逾期的用户，剩下用户直接统计还款日作为结果) 二分类或多分类（逾期，逾期前1天、2天、3天还款）
    args = parse_command_params()

    # analysis_data(args['path'])

    train, test = gen_data(args)
    cols = list(set(train.columns) - set(['auditing_date', 'due_date', 'insertdate', 'label']))
    model = models.fit(train, args, cols)
    result = gen_result(model, test, cols)
    save_zip(result, args)

    print('done !')


