from __future__ import absolute_import, division, print_function
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from time import gmtime, strftime

from config import logger, config
from feature import get_train_test_features, get_train_test_features2, get_train_test_features3, get_train_test_features4, get_train_test_features2a

def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((config.n_class, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True


def submit_result(submit, result):
    submit['recommend_mode'] = result
    submit.to_csv(config.trn_bag_submission_file, index=False)
    
def train_lgb(trn, y, tst):
    params = {'objective': 'multiclass', 
              'num_class': 12, 
              'seed': 2019, 
              'learning_rate': 0.05, 
              'num_threads': 8, 
              'num_leaves': 44, 
              'max_depth': 11, 
              'lambda_l1': 4.717461111446621, 
              'lambda_l2': 10.550885244591129, 
              'feature_fraction': 0.8235898660709667, 
              'bagging_fraction': 0.9018152298305773, 
              'bagging_freq': 3,
              'verbose': -1}
    
    cat_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                'min_price_mode', 'max_eta_mode', 'min_eta_mode',
                'first_mode', 'weekday', 'hour']

    p = np.zeros_like(y)
    best_iteration = 927
        
    lgb_trn = lgb.Dataset(trn, y, categorical_feature=cat_cols, free_raw_data=False)
    prob_trn_tst = 0
    for seed in [0, 17, 23, 29]:
        params['seed'] = 2019 + seed
        print(params)
        clf = lgb.train(params, lgb_trn,
                        valid_sets=[lgb_trn],
                        num_boost_round=best_iteration,
                        verbose_eval=50,
                        feval=eval_f)
        
        prob_trn_tst += clf.predict(tst)
    
    prob_trn_tst /= 4.0

    np.savetxt(config.predict_trn_tst_bag_file, prob_trn_tst, delimiter=',')
    
    trn_tst = np.argmax(prob_trn_tst, axis=1)

    return trn_tst

if __name__ == '__main__':

    trn, y, tst, sub = get_train_test_features2a()

    config.set_algo_name('lgb3')
    config.set_feature_name('f2a')
    p_tst = train_lgb(trn, y, tst)

    submit_result(sub, p_tst)