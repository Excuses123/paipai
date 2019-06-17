import lightgbm as lgb
import xgboost as xgb
import pandas as pd


def fit(train, args, cols):

    if args['model'] == 'lgb':

        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 127,
            'learning_rate': args['learning_rate'],
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_seed': 0,
            'bagging_freq': 1,
            'verbose': 1,
            'reg_alpha': 1,
            'reg_lambda': 2
        }

        lgb_train = lgb.Dataset(train[cols], train['label'])

        lgb_bst = lgb.cv(lgb_params, lgb_train, nfold=5, num_boost_round=int(args['round']), early_stopping_rounds=100, verbose_eval=50, stratified=False)
        print('best round = ',len(lgb_bst['rmse-mean']))
        model = lgb.train(lgb_params, lgb_train, num_boost_round=len(lgb_bst['rmse-mean']))

        lgb_importance = pd.DataFrame(
            {'gain': list(model.feature_importance(importance_type='gain')), 'feature': cols})
        print(lgb_importance.sort_values(by=['gain'], ascending=False).head(30))

    return model