# -*- coding: utf-8 -*-


import json
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def read_profile_data():
    profile_data = pd.read_csv('../input/profiles.csv')
    profile_na = np.zeros(67)
    profile_na[0] = -1
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))
    profile_na.columns = profile_data.columns
    profile_data = profile_data.append(profile_na)
    return profile_data


def merge_raw_data():
    tr_queries = pd.read_csv('../input/train_queries.csv')
    te_queries = pd.read_csv('../input/test_queries.csv')
    tr_plans = pd.read_csv('../input/train_plans.csv')
    te_plans = pd.read_csv('../input/test_plans.csv')

    tr_click = pd.read_csv('../input/train_clicks.csv')

    tr_data = tr_queries.merge(tr_click, on='sid', how='left')
    tr_data = tr_data.merge(tr_plans, on='sid', how='left')
    tr_data = tr_data.drop(['click_time'], axis=1)
    tr_data['click_mode'] = tr_data['click_mode'].fillna(0)

    te_data = te_queries.merge(te_plans, on='sid', how='left')
    te_data['click_mode'] = -1

    data = pd.concat([tr_data, te_data], axis=0)
    data = data.drop(['plan_time'], axis=1)
    data = data.reset_index(drop=True)
    print('total data size: {}'.format(data.shape))
    print('raw data columns: {}'.format(', '.join(data.columns)))
    return data


def gen_od_feas(data):
    enc = LabelEncoder()
    data['o1'] = data['o'].apply(lambda x: float(x.split(',')[0]))
    data['o2'] = data['o'].apply(lambda x: float(x.split(',')[1]))
    data['d1'] = data['d'].apply(lambda x: float(x.split(',')[0]))
    data['d2'] = data['d'].apply(lambda x: float(x.split(',')[1]))
    
    #data['o_enc'] = enc.fit_transform(data['o'])
    #data['d_enc'] = enc.fit_transform(data['d'])
    data = data.drop(['o', 'd'], axis=1)
    return data


def gen_plan_feas(data):
    n = data.shape[0]
    mode_list_feas = np.zeros((n, 12))
    max_dist, min_dist, mean_dist, std_dist = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_price, min_price, mean_price, std_price = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_eta, min_eta, mean_eta, std_eta = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    min_dist_mode, max_dist_mode, min_price_mode, max_price_mode, min_eta_mode, max_eta_mode, first_mode = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    mode_texts = []
    for i, plan in tqdm(enumerate(data['plans'].values)):
        try:
            cur_plan_list = json.loads(plan)
        except:
            cur_plan_list = []
        if len(cur_plan_list) == 0:
            mode_list_feas[i, 0] = 1
            first_mode[i] = 0

            max_dist[i] = -1
            min_dist[i] = -1
            mean_dist[i] = -1
            std_dist[i] = -1

            max_price[i] = -1
            min_price[i] = -1
            mean_price[i] = -1
            std_price[i] = -1

            max_eta[i] = -1
            min_eta[i] = -1
            mean_eta[i] = -1
            std_eta[i] = -1

            min_dist_mode[i] = -1
            max_dist_mode[i] = -1
            min_price_mode[i] = -1
            max_price_mode[i] = -1
            min_eta_mode[i] = -1
            max_eta_mode[i] = -1

            mode_texts.append('word_null')
        else:
            distance_list = []
            price_list = []
            eta_list = []
            mode_list = []
            for tmp_dit in cur_plan_list:
                distance_list.append(int(tmp_dit['distance']))
                if tmp_dit['price'] == '':
                    price_list.append(0)
                else:
                    price_list.append(int(tmp_dit['price']))
                eta_list.append(int(tmp_dit['eta']))
                mode_list.append(int(tmp_dit['transport_mode']))
            mode_texts.append(
                ' '.join(['word_{}'.format(mode) for mode in mode_list]))
            distance_list = np.array(distance_list)
            price_list = np.array(price_list)
            eta_list = np.array(eta_list)
            mode_list = np.array(mode_list, dtype='int')
            mode_list_feas[i, mode_list] = 1
            distance_sort_idx = np.argsort(distance_list)
            price_sort_idx = np.argsort(price_list)
            eta_sort_idx = np.argsort(eta_list)

            max_dist[i] = distance_list[distance_sort_idx[-1]]
            min_dist[i] = distance_list[distance_sort_idx[0]]
            mean_dist[i] = np.mean(distance_list)
            std_dist[i] = np.std(distance_list)

            max_price[i] = price_list[price_sort_idx[-1]]
            min_price[i] = price_list[price_sort_idx[0]]
            mean_price[i] = np.mean(price_list)
            std_price[i] = np.std(price_list)

            max_eta[i] = eta_list[eta_sort_idx[-1]]
            min_eta[i] = eta_list[eta_sort_idx[0]]
            mean_eta[i] = np.mean(eta_list)
            std_eta[i] = np.std(eta_list)

            first_mode[i] = mode_list[0]
            max_dist_mode[i] = mode_list[distance_sort_idx[-1]]
            min_dist_mode[i] = mode_list[distance_sort_idx[0]]

            max_price_mode[i] = mode_list[price_sort_idx[-1]]
            min_price_mode[i] = mode_list[price_sort_idx[0]]

            max_eta_mode[i] = mode_list[eta_sort_idx[-1]]
            min_eta_mode[i] = mode_list[eta_sort_idx[0]]

    feature_data = pd.DataFrame(mode_list_feas)
    feature_data.columns = ['mode_feas_{}'.format(i) for i in range(12)]
    feature_data['max_dist'] = max_dist
    feature_data['min_dist'] = min_dist
    feature_data['mean_dist'] = mean_dist
    feature_data['std_dist'] = std_dist

    feature_data['max_price'] = max_price
    feature_data['min_price'] = min_price
    feature_data['mean_price'] = mean_price
    feature_data['std_price'] = std_price

    feature_data['max_eta'] = max_eta
    feature_data['min_eta'] = min_eta
    feature_data['mean_eta'] = mean_eta
    feature_data['std_eta'] = std_eta

    feature_data['max_dist_mode'] = max_dist_mode
    feature_data['min_dist_mode'] = min_dist_mode
    feature_data['max_price_mode'] = max_price_mode
    feature_data['min_price_mode'] = min_price_mode
    feature_data['max_eta_mode'] = max_eta_mode
    feature_data['min_eta_mode'] = min_eta_mode
    feature_data['first_mode'] = first_mode
    print('mode tfidf...')
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(mode_texts)
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_mode_{}'.format(i) for i in range(10)]

    data = pd.concat([data, feature_data, mode_svd], axis=1)
    data = data.drop(['plans'], axis=1)
    return data


def gen_profile_feas(data):
    profile_data = read_profile_data()
    x = profile_data.drop(['pid'], axis=1).values
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=2019)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    svd_feas.columns = ['svd_fea_{}'.format(i) for i in range(20)]
    svd_feas['pid'] = profile_data['pid'].values
    data['pid'] = data['pid'].fillna(-1)
    data = data.merge(svd_feas, on='pid', how='left')
    return data


def group_weekday_and_hour(row):
    if row['weekday'] == 0 or row['weekday'] == 6:
        w = 0 
    else:
        w = row['weekday']
    if row['hour'] > 7 and row['hour'] < 18: # 7:00 - 18:00
        h = row['hour'] 
    elif row['hour'] >= 18 and row['hour'] < 21: # 18:00 - 21:00
        h = 1
    elif row['hour'] >= 21 or row['hour'] < 6: # 21:00 - 6:00
        h = 0
    else: # 6:00 - 7:00
        h = 2
    
    return str(w) + '_' + str(h)

def gen_time_feas(data):
    data['req_time'] = pd.to_datetime(data['req_time'])
    data['weekday'] = data['req_time'].dt.dayofweek
    data['hour'] = data['req_time'].dt.hour
    #data['group-weekday-n-hour'] = LabelEncoder().fit_transform(data.apply(group_weekday_and_hour, axis = 1))
    data = data.drop(['req_time'], axis=1)
    return data

def gen_ratio_feas(data):
    data['dist-d-eta'] = data['mean_dist'] / data['mean_eta']
    data['price-d-dist'] = data['mean_price'] / data['mean_dist']
    data['price-d-eta'] = data['mean_price'] / data['mean_eta']
    data['o1-d-d1'] = data['o1'] / data['d1']
    data['o2-d-d2'] = data['o2'] / data['d2']
    return data

def gen_fly_dist_feas(data):
    data['fly-dist'] = ((data['d1'] - data['o1'])**2 + (data['d2'] - data['o2'])**2)**0.5
    data['fly-dist-d-dist'] = data['fly-dist'] / data['mean_dist']
    data['fly-dist-d-eta'] = data['fly-dist'] / data['mean_eta']
    data['price-d-fly-dist'] = data['mean_price'] / data['fly-dist']
    return data

def gen_aggregate_profile_feas(data):
    aggr = data.groupby('pid')['sid'].agg(['count'])  
    aggr.columns = ['%s_%s' % ('sid', col) for col in aggr.columns.values]
    aggr = aggr.reset_index()
    aggr.loc[aggr['pid'] == -1.0,'sid_count'] = 0 # reset in case pid == -1
    data = data.merge(aggr, how='left', on=['pid'])
    return data

def gen_pid_feat(data):
    feat = pd.read_csv('../feat/pid_feat.csv')
    data = data.merge(feat, how='left', on='pid')
    return data

def gen_sid_feat(data):
    feat = pd.read_csv('../feat/sid_feat.csv')
    data = data.merge(feat, how='left', on='sid')
    data['first_mode-eq-min_dist_mode'] = (data['first_mode']==data['min_dist_mode']).astype(int)
    data['first_mode-eq-min_eta_mode'] = (data['first_mode']==data['min_eta_mode']).astype(int)
    data['first_mode-eq-min_price_mode'] = (data['first_mode']==data['min_price_mode']).astype(int)
    return data

def gen_od_feat(data):
    feat = pd.read_csv('../feat/od_feat.csv')
    tr_sid = pd.read_csv('../input/train_queries.csv', usecols=['sid','o','d'])
    te_sid = pd.read_csv('../input/test_queries.csv', usecols=['sid','o','d'])
    sid = pd.concat((tr_sid, te_sid))
    print(sid.shape)
    feat = sid.merge(feat, how='left', on=['o','d']).drop(['o','d'], axis=1)
    print(feat.shape)
    print(feat.columns)
    data = data.merge(feat, how='left', on='sid')
    return data
        
def gen_od_cluster_feat(data):
    feat = pd.read_csv('../feat/od_node_cluster.csv')
    tr_sid = pd.read_csv('../input/train_queries.csv', usecols=['sid','o','d'])
    te_sid = pd.read_csv('../input/test_queries.csv', usecols=['sid','o','d'])
    sid = pd.concat((tr_sid, te_sid))
    
    f = feat.copy()
    feat = sid.merge(feat, how='left', left_on='o', right_on='od').drop(['od','o'], axis=1)
    feat.rename(columns={'cluster': 'o_cluster'})
    feat = feat.merge(f, how='left', left_on='d', right_on='od').drop(['od','d'], axis=1)
    feat.rename(columns={'cluster': 'd_cluster'})
    
    data = data.merge(feat, how='left', on='sid')
    return data
    
def gen_od_eq_feat(data):
    data['o1-eq-d1'] = (data['o1'] == data['d1']).astype(int)
    data['o2-eq-d2'] = (data['o2'] == data['d2']).astype(int)
    data['o-eq-d'] = data['o1-eq-d1']*data['o2-eq-d2']
    return data
    
def gen_encode_feas(data):
    return data


def split_train_test(data):
    train_data = data[data['click_mode'] != -1]
    test_data = data[data['click_mode'] == -1]
    submit = test_data[['sid']].copy()
    train_data = train_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['click_mode'], axis=1)
    train_y = train_data['click_mode'].values
    train_x = train_data.drop(['click_mode'], axis=1)
    return train_x, train_y, test_data, submit


def get_train_test_feas_data():
    data = merge_raw_data()
    data = gen_od_feas(data)
    data = gen_plan_feas(data)
    data = gen_profile_feas(data)
    data = gen_time_feas(data) # 0.6758
    data = gen_ratio_feas(data)
    data = gen_fly_dist_feas(data)
    data = gen_aggregate_profile_feas(data) # 0.6759966661470926
    data = gen_pid_feat(data) # 0.6762996872664375
    #data = gen_sid_feat(data) # 0.6752915844109314 (not work)
    data = gen_od_feat(data) #  without click count: 0.6780576865566392; with click count: 0.6795810670221226
    data = gen_od_cluster_feat(data) # 0.6796523605372234
    data = gen_od_eq_feat(data)
    train_x, train_y, test_x, submit = split_train_test(data)
    return train_x, train_y, test_x, submit


if __name__ == '__main__':
    pass