from __future__ import absolute_import, division, print_function
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from time import gmtime, strftime

from config import logger, config
from feature import get_train_test_features, get_train_test_features2, get_train_test_features3


import numpy as np

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(12):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*12
  for i in range(12):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x

def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((config.n_class, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True

def f1_weighted(labels, preds):
    preds = np.argmax(preds.reshape(12, -1), axis=0)
    score = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_weighted', score, True

def to_label(x, p):
    p2 = np.zeros_like(p)
    for i in range(12):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    
    p2 = np.argmax(p2, axis=1)
    return p2

def f1_adj_weighted(labels, preds):
    x = [0.26, 0.41, 0.42, 0.18, 0.1, 0.45, 0.12, 0.39, 0.16, 0.33, 0.17, 0.29]
    preds = to_label(x, preds.reshape(12, -1).T)
    score = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_adj_weighted', score, True

def submit_result(submit, result, trn_result, score):
    now_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    submit['recommend_mode'] = result
    submit.to_csv(config.submission_file, index=False)
    
    if trn_result is not None:
        submit['recommend_mode'] = trn_result
        submit.to_csv(config.trn_submission_file, index=False)

    if os.path.exists(config.metric_file):
        metric = pd.read_csv(config.metric_file)
        metrics = metric.append({'model': config.model_name,
                                 'feature': config.feature_name,
                                 'datetime': now_time,
                                 'score': score}, ignore_index=True)
    else:
        metric = pd.DataFrame({'model': [config.model_name], 
                               'feature': config.feature_name,
                               'datetime': [now_time],
                               'score': [score]})

    metric.round(6).to_csv(config.metric_file, index=False)


def train_lgb(trn, y, tst=None):
    clf = lgb.LGBMClassifier(boosting_type="gbdt",
                               num_leaves=80,
                               reg_alpha=10,
                               reg_lambda=0.01,
                               max_depth=6,
                               n_estimators=2000,
                               objective='multiclass',
                               subsample=0.8,
                               colsample_bytree=0.8,
                               subsample_freq=1,
                               min_child_samples=50,
                               learning_rate=0.05,
                               metric="multiclass",
                               num_class=12,
                               feature_fraction=0.8,
                               bagging_fraction=0.8,
                               bagging_freq=4,
                               n_jobs=-1, 
                               seed=2019,
                               verbose=-1)
    
    
    cat_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                'min_price_mode', 'max_eta_mode', 'min_eta_mode',
                'first_mode', 'weekday', 'hour', 'weather']

    cat_cols = ['max_dist_mode', 'min_dist_mode', 'max_price_mode',
                'min_price_mode', 'max_eta_mode', 'min_eta_mode',
                'first_mode', 'weekday', 'hour']

    #cat_cols = ['pid', 'max_dist_mode', 'min_dist_mode', 'max_price_mode',
    #            'min_price_mode', 'max_eta_mode', 'min_eta_mode', 'first_mode', 'weekday', 'hour', 
    #            ]
                
    X_trn, y_trn, X_val, y_val = trn.iloc[:-63388,:], y[:-63388], trn.iloc[-63388:,], y[-63388:]
    
    eval_set = [(X_trn, y_trn), (X_val, y_val)]
    clf.fit(X_trn, y_trn, eval_set=eval_set, eval_metric=f1_weighted, categorical_feature=cat_cols, verbose=10, early_stopping_rounds=100)
    #clf.fit(X_trn, y_trn, eval_set=eval_set, eval_metric=f1_adj_weighted, categorical_feature=cat_cols, verbose=10, early_stopping_rounds=100)    

    feature_importances = list(clf.feature_importances_)
    feature_names = trn.columns.values.tolist()
    imp = pd.DataFrame({'feature_importances': feature_importances, 'feature_names':feature_names})
    imp = imp.sort_values('feature_importances', ascending=False).drop_duplicates()
    print("[+] All feature importances", list(imp.values))

    pred = clf.predict(X_val, num_iteration=clf.best_iteration_)
    print('Val F1: %f',  f1_score(y_val, pred, average='weighted'))
    print(classification_report(y_val, pred))

if __name__ == '__main__':

    trn, y, tst, sub = get_train_test_features2()
    #df = pd.read_csv(config.train_feature_file)
    #df = df[~pd.isnull(df['click_mode'])]

    #trn = df.drop(['sid','req_time', 'click_mode'], axis=1)
    #y = df['click_mode'].values

    config.set_algo_name('lgb4')
    config.set_feature_name('f2') # f2 = 
    train_lgb(trn, y)

    