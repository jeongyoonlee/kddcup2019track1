import numpy as np
import pandas as pd

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
                       'feature': config.feature_name,
                       'datetime': now_time,
                       'score': score}, ignore_index=True)
    else:
        metric = pd.DataFrame({'model': [config.model_name], 
                               'feature': config.feature_name,
                               'datetime': [now_time],
                               'score': [score]})

    metric.round(6).to_csv(config.metric_file, index=False)