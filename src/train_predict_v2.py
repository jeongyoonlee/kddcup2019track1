from __future__ import absolute_import, division, print_function
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from time import gmtime, strftime

from config import logger, config
from feature import get_train_test_features, get_train_test_features2


def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((config.n_class, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True


def submit_result(submit, result, trn_result, score):
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    submit['recommend_mode'] = result
    submit.to_csv(config.submission_file, index=False)
    
    if trn_result is not None:
        submit['recommend_mode'] = trn_result
        submit.to_csv(config.trn_submission_file, index=False)

    if os.path.exists(config.metric_file):
        metric = pd.read_csv(config.metric_file)
        metric.append({'model': config.model_name,
                       'datetime': now_time,
                       'score': score}, ignore_index=True)
    else:
        metric = pd.DataFrame({'model': [config.model_name],
                               'datetime': [now_time],
                               'score': [score]})

    metric.round(6).to_csv(config.metric_file, index=False)


def train_lgb(trn, y, tst):
    cv_idx = np.loadtxt(config.cv_id_file)
    params = {'objective': 'multiclass', 
              'num_class': 12, 
              'seed': 2019, 
              'learning_rate': 0.05, 
              'num_threads': 4, 
              'num_leaves': 39, 
              'max_depth': 7, 
              'lambda_l1': 3.189387005111133, 
              'lambda_l2': 13.207486031673932, 
              'feature_fraction': 0.5991759690382027, 
              'bagging_fraction': 0.9806964428838549, 
              'bagging_freq': 3,
              'verbose': -1}

    cat_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                'min_price_mode', 'max_eta_mode', 'min_eta_mode',
                'first_mode', 'weekday', 'hour']
    p = np.zeros_like(y)
    prob = np.zeros((trn.shape[0], config.n_class), dtype=float)
    prob_tst = np.zeros((tst.shape[0], config.n_class))
    best_iteration = 519
    for k, _id in enumerate(range(5)):
        i_trn = (cv_idx != _id)
        i_val = (cv_idx == _id)
        print(' [*] Train id: ', i_trn)
        print(' [*] Valid id: ', i_val)
        X_trn, y_trn, X_val, y_val = trn.iloc[i_trn], y[i_trn], trn.iloc[i_val], y[i_val]
        lgb_trn = lgb.Dataset(X_trn, y_trn, categorical_feature=cat_cols)
        lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols)
        clf = lgb.train(params, lgb_trn,
                        valid_sets=[lgb_trn, lgb_val],
                        num_boost_round=best_iteration,
                        verbose_eval=50,
                        feval=eval_f)
        prob[i_val,:] = clf.predict(X_val)
        p[i_val] = np.argmax(prob[i_val], axis=1)
        score = f1_score(y_val, p[i_val], average='weighted')
        prob_tst += clf.predict(tst) / config.n_fold
        print('[+] fold #{}: {}'.format(k, score))
        feature_importances = list(clf.feature_importance())
        # print('Feature importances:', feature_importances)
        # review feature importances
        feature_names = trn.columns.values.tolist()
        imp = pd.DataFrame({'feature_importances': feature_importances, 'feature_names':feature_names})
        imp = imp.sort_values('feature_importances', ascending=False).drop_duplicates()
        print("[+] All feature importances", list(imp.values))
    
    lgb_trn = lgb.Dataset(trn, y, categorical_feature=cat_cols)
    clf = lgb.train(params, lgb_trn,
                    valid_sets=[lgb_trn],
                    num_boost_round=best_iteration,
                    verbose_eval=50,
                    feval=eval_f)
    
    prob_trn_tst = clf.predict(tst)
    
    imp.to_csv(config.feature_imp_file, index=False)
    score = f1_score(y, p, average='weighted')
    print('[+] CV f1-score: ', score)
    p_tst = np.argmax(prob_tst, axis=1)
    p_trn_tst = np.argmax(prob_trn_tst, axis=1)
    
    np.savetxt(config.predict_val_file, prob, delimiter=',')
    np.savetxt(config.predict_tst_file, prob_tst, delimiter=',')
    np.savetxt(config.predict_trn_tst_file, prob_trn_tst, delimiter=',')
    
    return p_tst, p_trn_tst, score


if __name__ == '__main__':

    trn, y, tst, sub = get_train_test_features()

    config.set_algo_name('lgb2')
    p_tst, p_trn_tst, score = train_lgb(trn, y, tst)

    submit_result(sub, p_tst, p_trn_tst, score)