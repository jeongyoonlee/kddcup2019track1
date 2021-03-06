# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb

import build_features

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from time import gmtime, strftime


def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((12, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True


def submit_result(submit, result, score, model_name):
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    submit['recommend_mode'] = result
    submit.to_csv(
        '../submit/{}_result_{}_{}.csv'.format(model_name, score, now_time), index=False)


def train_lgb(train_x, train_y, test_x):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    lgb_paras = {
        'objective': 'multiclass',
        'metrics': 'multiclass',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'lambda_l1': 0.01,
        'lambda_l2': 10,
        'num_class': 12,
        'seed': 2019,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
    }
    cate_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                 'min_price_mode', 'max_eta_mode', 'min_eta_mode', 
                 'first_mode', 'weekday', 'hour']
    scores = []
    result_proba = []
    for k, (tr_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        tr_x, tr_y, val_x, val_y = train_x.iloc[tr_idx], train_y[tr_idx], train_x.iloc[val_idx], train_y[val_idx]
        train_set = lgb.Dataset(tr_x, tr_y, categorical_feature=cate_cols)
        val_set = lgb.Dataset(val_x, val_y, categorical_feature=cate_cols)
        lgb_model = lgb.train(lgb_paras, train_set,
                              valid_sets=[val_set], early_stopping_rounds=50, num_boost_round=40000, verbose_eval=50, feval=eval_f)
        val_pred = np.argmax(lgb_model.predict(
            val_x, num_iteration=lgb_model.best_iteration), axis=1)
        val_score = f1_score(val_y, val_pred, average='weighted')
        result_proba.append(lgb_model.predict(
            test_x, num_iteration=lgb_model.best_iteration))
        scores.append(val_score)
        print('[+] fold #{}: {}'.format(k, val_score))
        feature_importances = list(lgb_model.feature_importance())
        # print('Feature importances:', feature_importances)
        # review feature importances
        feature_names = train_x.columns.values.tolist()
        df = pd.DataFrame({'feature_importances': feature_importances, 'feature_names':feature_names})
        df = df.sort_values('feature_importances', ascending=False).drop_duplicates()
        print("[+] All feature importances", list(df.values))
        
    print('[+] cv f1-score: ', np.mean(scores))
    pred_test = np.argmax(np.mean(result_proba, axis=0), axis=1)
    return pred_test, np.mean(scores)


if __name__ == '__main__':
    train_x, train_y, test_x, submit = build_features.get_train_test_feas_data()
    result_lgb, score = train_lgb(train_x, train_y, test_x)
    submit_result(submit, result_lgb, score, 'lgb')