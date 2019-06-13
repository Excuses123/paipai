import lightgbm as lgb
import xgboost as xgb




def fit(train):

    model = lgb.train(train)

    return model