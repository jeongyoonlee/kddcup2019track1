
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


if __name__ == '__main__':
    cv = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
    trn, y, tst, sub = get_train_test_features()
    cv_idx = np.zeros_like(y)
    for k, (i_trn, i_val) in enumerate(cv.split(trn, y)):
        cv_idx[i_val] = k
    
    np.savetxt(config.cv_id_file, cv_idx, fmt='%d')
