import json
import networkx as nx
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from config import logger, config


def read_profile_data():
    profile_na = np.zeros(67)
    profile_na[0] = -1
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))

    profile_df = pd.read_csv(config.profile_file)
    profile_na.columns = profile_df.columns
    profile_df = profile_df.append(profile_na)
    return profile_df


def merge_raw_data():
    tr_queries = pd.read_csv(config.train_query_file, parse_dates=['req_time'])
    te_queries = pd.read_csv(config.test_query_file, parse_dates=['req_time'])
    tr_plans = pd.read_csv(config.train_plan_file, parse_dates=['plan_time'])
    te_plans = pd.read_csv(config.test_plan_file, parse_dates=['plan_time'])

    tr_click = pd.read_csv(config.train_click_file)

    trn = tr_queries.merge(tr_click, on='sid', how='left')
    trn = trn.merge(tr_plans, on='sid', how='left')
    trn = trn.drop(['click_time'], axis=1)
    trn['click_mode'] = trn['click_mode'].fillna(0)

    tst = te_queries.merge(te_plans, on='sid', how='left')
    tst['click_mode'] = -1

    df = pd.concat([trn, tst], axis=0, sort=False)
    df = df.drop(['plan_time'], axis=1)
    df = df.reset_index(drop=True)

    df['weekday'] = df['req_time'].dt.weekday
    df['day'] = df['req_time'].dt.day
    df['hour'] = df['req_time'].dt.hour
    df = df.drop(['req_time'], axis=1)

    logger.info('total data size: {}'.format(df.shape))
    logger.info('data columns: {}'.format(', '.join(df.columns)))

    return df


def extract_plans(df):
    plans = []
    for sid, plan in tqdm(zip(df['sid'].values, df['plans'].values)):
        try:
            p = json.loads(plan)
            for x in p:
                x['sid'] = sid

            plans.extend(p)
        except:
            pass

    return pd.DataFrame(plans)


def generate_od_features(df):
    feat = df[['o','d']].drop_duplicates()
    feat = feat.merge(df.groupby('o')[['day', 'hour', 'pid', 'click_mode']].nunique().reset_index(), how='left', on='o')
    feat.rename(columns={'day': 'o_nunique_day',
                         'hour': 'o_nunique_hour',
                         'pid': 'o_nunique_pid',
                         'click_mode': 'o_nunique_click'}, inplace=True)

    feat = feat.merge(df.groupby('d')[['day', 'hour', 'pid', 'click_mode']].nunique().reset_index(), how='left', on='d')
    feat.rename(columns={'day': 'd_nunique_day',
                         'hour': 'd_nunique_hour',
                         'pid': 'd_nunique_pid',
                         'click_mode': 'd_nunique_click'}, inplace=True)

    feat = feat.merge(df.groupby(['o', 'd'])[['day', 'hour', 'pid', 'click_mode']].nunique().reset_index(), how='left', on=['o', 'd'])
    feat.rename(columns={'day': 'od_nunique_day',
                         'hour': 'od_nunique_hour',
                         'pid': 'od_nunique_pid',
                         'click_mode': 'od_nunique_click'}, inplace=True)

    return feat


def generate_pid_features(df):
    feat = df.groupby('pid')[['hour', 'day']].nunique().reset_index()
    feat.rename(columns={'hour': 'pid_nunique_hour', 'day': 'pid_nunique_day'}, inplace=True)
    feat['nunique_hour_d_nunique_day'] = feat['pid_nunique_hour'] / feat['pid_nunique_day']

    feat = feat.merge(df.groupby('pid')[['o', 'd']].nunique().reset_index(), how='left', on='pid')
    feat.rename(columns={'o': 'pid_nunique_o', 'd': 'pid_nunique_d'}, inplace=True)
    feat['nunique_o_d_nunique_d'] = feat['pid_nunique_o'] / feat['pid_nunique_d']

    return feat


def generate_od_cluster_features(df):
    G = nx.Graph()
    G.add_nodes_from(df['o'].unique().tolist())
    G.add_nodes_from(df['d'].unique().tolist())

    edges = df[['o','d']].apply(lambda x: (x[0],x[1]), axis=1).tolist()
    G.add_edges_from(edges)
    cluster = nx.clustering(G)

    cluster_df = pd.DataFrame([{'od': key, 'cluster': cluster[key]} for key in cluster.keys()])
    return cluster_df


def gen_od_feas(data):
    data['o1'] = data['o'].apply(lambda x: float(x.split(',')[0]))
    data['o2'] = data['o'].apply(lambda x: float(x.split(',')[1]))
    data['d1'] = data['d'].apply(lambda x: float(x.split(',')[0]))
    data['d2'] = data['d'].apply(lambda x: float(x.split(',')[1]))

    data = data.drop(['o', 'd'], axis=1)
    return data


def gen_plan_feas(data):
    n = data.shape[0]
    mode_list_feas = np.zeros((n, 12))
    max_dist, min_dist, mean_dist, std_dist = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    max_price, min_price, mean_price, std_price = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    max_eta, min_eta, mean_eta, std_eta = np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
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
    logger.info('mode tfidf...')
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
    feat = pd.read_csv(config.pid_feature_file)
    data = data.merge(feat, how='left', on='pid')
    return data

def gen_od_feat(data):
    feat = pd.read_csv(config.od_feature_file)
    tr_sid = pd.read_csv(config.train_query_file, usecols=['sid','o','d'])
    te_sid = pd.read_csv(config.test_query_file, usecols=['sid','o','d'])
    sid = pd.concat((tr_sid, te_sid))
    logger.info('sid shape={}'.format(sid.shape))
    feat = sid.merge(feat, how='left', on=['o','d']).drop(['o','d'], axis=1)
    logger.info('feature shape={}'.format(feat.shape))
    logger.info('feature columns={}'.format(feat.columns))
    data = data.merge(feat, how='left', on='sid')
    click_cols = [c for c in feat.columns if c.endswith('click')]
    data.drop(click_cols, axis=1, inplace=True)
    return data

def gen_od_cluster_feat(data):
    feat = pd.read_csv(config.od_cluster_feature_file)
    tr_sid = pd.read_csv(config.train_query_file, usecols=['sid','o','d'])
    te_sid = pd.read_csv(config.test_query_file, usecols=['sid','o','d'])
    sid = pd.concat((tr_sid, te_sid))

    f = feat.copy()
    feat = sid.merge(feat, how='left', left_on='o', right_on='od').drop(['od','o'], axis=1)
    feat.rename(columns={'cluster': 'o_cluster'}, inplace=True)
    feat = feat.merge(f, how='left', left_on='d', right_on='od').drop(['od','d'], axis=1)
    feat.rename(columns={'cluster': 'd_cluster'}, inplace=True)

    data = data.merge(feat, how='left', on='sid')
    return data

def gen_od_eq_feat(data):
    data['o1-eq-d1'] = (data['o1'] == data['d1']).astype(int)
    data['o2-eq-d2'] = (data['o2'] == data['d2']).astype(int)
    data['o-eq-d'] = data['o1-eq-d1']*data['o2-eq-d2']

    data['o1-m-o2'] = np.abs(data['o1'] - data['o2'])
    data['d1-m-d2'] = np.abs(data['d1'] - data['d2'])

    data['od_area'] = data['o1-m-o2']*data['d1-m-d2']
    data['od_ratio'] = data['o1-m-o2']/data['d1-m-d2']

    return data

def gen_od_mode_cnt_feat(data):
    feat = pd.read_csv(config.od_mode_cnt_feature_file)
    tr_sid = pd.read_csv(config.train_query_file, usecols=['sid','o','d'])
    te_sid = pd.read_csv(config.test_query_file, usecols=['sid','o','d'])
    sid = pd.concat((tr_sid, te_sid))

    feat = sid.merge(feat, how='left', on=['o','d']).drop(['o','d'], axis=1)
    data = data.merge(feat, how='left', on='sid')
    return data

def gen_weekday_hour_cnt_feat(data):
    feat = pd.read_csv(config.weekday_hour_feature_file)
    tr_sid = pd.read_csv(config.train_query_file, usecols=['sid','req_time'])
    te_sid = pd.read_csv(config.test_query_file, usecols=['sid','req_time'])
    sid = pd.concat((tr_sid, te_sid))
    sid['req_time'] = pd.to_datetime(sid['req_time'])
    sid['hour'] = sid['req_time'].map(lambda x: x.hour)
    sid['weekday'] = sid['req_time'].map(lambda x: x.weekday())

    feat = sid.merge(feat, how='left', on=['hour','weekday']).drop(['hour','weekday','req_time'], axis=1)
    data = data.merge(feat, how='left', on='sid')
    return data

def gen_od_plan_agg_feat(data):
    #feat = pd.read_csv(config.od_plan_agg_feature_file)
    #tr_sid = pd.read_csv(config.train_query_file, usecols=['sid','o','d','req_time'])
    #te_sid = pd.read_csv(config.test_query_file, usecols=['sid','o','d', 'req_time'])
    #sid = pd.concat((tr_sid, te_sid))
    #sid['req_time'] = pd.to_datetime(sid['req_time'])
    #sid['hour'] = sid['req_time'].map(lambda x: x.hour)

    #feat = sid.merge(feat, how='left', on=['o','d','hour']).drop(['o','d','hour','req_time'], axis=1)
    feat = pd.read_csv(config.od_plan_agg_feature_file)
    data = data.merge(feat, how='left', on='sid')
    return data   

def gen_mode_feat(data):
    feat = pd.read_csv(config.mode_feature_file)
    data = data.merge(feat, how='left', on='sid')
    return data

def gen_mode_stats_feat(data):
    feat = pd.read_csv(config.od_stats_file)
    data = data.merge(feat, how='left', on='sid')
    return data

def gen_daily_plan_feat(data):
    feat = pd.read_csv(config.daily_plan_file)
    data = data.merge(feat, how='left', on='sid')
    return data

def gen_weather_feat(data):
    feat = pd.read_csv(config.weather_file)
    data = data.merge(feat, how='left', on='sid')
    return data

def gen_od_pid_count_feat(data):
    feat = pd.read_csv(config.od_pid_count_file)
    data = data.merge(feat, how='left', on='sid')
    return data

def gen_plan_ratio_feat(data):
    feat = pd.read_csv(config.plan_ratio_file)
    data = data.merge(feat, how='left', on='sid')
    return data

def generate_f1(df):
    trn_feat_name, tst_feat_name = config.get_feature_name('f1')
    if os.path.exists(trn_feat_name) and os.path.exists(tst_feat_name):
        logger.info('loading the training and test features from files.')
        trn = pd.read_csv(trn_feat_name)
        tst = pd.read_csv(tst_feat_name)
    else:
        df = gen_od_feas(df)
        df = gen_plan_feas(df)
        df = gen_profile_feas(df)
        df = gen_ratio_feas(df)
        df = gen_fly_dist_feas(df)
        df = gen_aggregate_profile_feas(df) # 0.6759966661470926
        df = gen_pid_feat(df) # 0.6762996872664375
        df = gen_od_feat(df) #  without click count: 0.6780576865566392; with click count: 0.6795810670221226
        df = gen_od_cluster_feat(df) # 0.6796523605372234
        df = gen_od_eq_feat(df)

        trn = df[df['click_mode'] != -1]
        tst = df[df['click_mode'] == -1]

    return trn, tst

def generate_f2(df):
    trn_feat_name, tst_feat_name = config.get_feature_name('f2')
    if os.path.exists(trn_feat_name) and os.path.exists(tst_feat_name):
        logger.info('loading the training and test features from files.')
        trn = pd.read_csv(trn_feat_name)
        tst = pd.read_csv(tst_feat_name)
    else:
        trn, tst = generate_f1(df)
        df = pd.concat((trn, tst))

        df = gen_od_mode_cnt_feat(df) # [+] fold #0: 0.6835031183515229
        df = gen_weekday_hour_cnt_feat(df)
        df = gen_od_plan_agg_feat(df)
        df = gen_mode_feat(df)

        #df = gen_mode_stats_feat(df)
        ## df = gen_weather_feat(df)
        #df = gen_daily_plan_feat(df)
        #df = gen_od_pid_count_feat(df)
        ## df = gen_plan_ratio_feat(df)

        trn = df[df['click_mode'] != -1]
        tst = df[df['click_mode'] == -1]

    return trn, tst


def generate_f3(df):
    trn_feat_name, tst_feat_name = config.get_feature_name('f1')
    if os.path.exists(trn_feat_name) and os.path.exists(tst_feat_name):
        logger.info('loading the training and test features from files.')
        trn = pd.read_csv(trn_feat_name)
        tst = pd.read_csv(tst_feat_name)
    else:
        trn, tst = generate_f2(df)
        df = pd.concat((trn, tst))


        #df = gen_mode_stats_feat(df)
        ## df = gen_weather_feat(df)
        #df = gen_daily_plan_feat(df)
        #df = gen_od_pid_count_feat(df)
        ## df = gen_plan_ratio_feat(df)

        trn = df[df['click_mode'] != -1]
        tst = df[df['click_mode'] == -1]

    return trn, tst

def get_train_test_features():
    config.set_feature_name('f1')
    if os.path.exists(config.train_feature_file) and os.path.exists(config.test_feature_file):
        logger.info('loading the training and test features from files.')
        trn = pd.read_csv(config.train_feature_file)
        tst = pd.read_csv(config.test_feature_file)
    else:
        df = merge_raw_data()
        logger.info('generating feature f1.')
        trn, tst = generate_f1(df)

        logger.info('saving the training and test f1 features.')
        trn.to_csv(config.train_feature_file, index=False)
        tst.to_csv(config.test_feature_file, index=False)

    y = trn['click_mode'].values
    sub = tst[['sid']].copy()

    trn.drop(['sid', 'pid', 'click_mode'], axis=1, inplace=True)
    tst.drop(['sid', 'pid', 'click_mode'], axis=1, inplace=True)

    return trn, y, tst, sub

def get_train_test_features2():
    config.set_feature_name('f2')
    if os.path.exists(config.train_feature_file) and os.path.exists(config.test_feature_file):
        logger.info('loading the training and test features from files.')
        trn = pd.read_csv(config.train_feature_file)
        tst = pd.read_csv(config.test_feature_file)
    else:
        df = merge_raw_data()
        logger.info('generating feature f2.')
        trn, tst = generate_f2(df)

        logger.info('saving the training and test f2 features.')
        trn.to_csv(config.train_feature_file, index=False)
        tst.to_csv(config.test_feature_file, index=False)

    y = trn['click_mode'].values
    sub = tst[['sid']].copy()

    trn.drop(['sid', 'pid', 'click_mode'], axis=1, inplace=True)
    tst.drop(['sid', 'pid', 'click_mode'], axis=1, inplace=True)

    return trn, y, tst, sub

def get_train_test_features2a():
    config.set_feature_name('f2')
    if os.path.exists(config.train_feature_file) and os.path.exists(config.test_feature_file):
        logger.info('loading the training and test features from files.')
        trn = pd.read_csv(config.train_feature_file)
        tst = pd.read_csv(config.test_feature_file)
    else:
        df = merge_raw_data()
        logger.info('generating feature f2.')
        trn, tst = generate_f2(df)

        logger.info('saving the training and test f2 features.')
        trn.to_csv(config.train_feature_file, index=False)
        tst.to_csv(config.test_feature_file, index=False)

    y = trn['click_mode'].values
    sub = tst[['sid']].copy()

    feat = pd.read_csv('/home/ubuntu/projects/kddcup2019track1/build/feature/od_coord_feature.csv')
    trn = trn.merge(feat, how='left', on='sid')
    tst = tst.merge(feat, how='left', on='sid')

    feat = pd.read_csv('/home/ubuntu/projects/kddcup2019track1/input/data_set_phase1/var_dist_time.csv')
    trn = trn.merge(feat, how='left', on='sid')
    tst = tst.merge(feat, how='left', on='sid')

    feat = pd.read_csv('/home/ubuntu/projects/kddcup2019track1/input/data_set_phase1/var_dist_min.csv')
    trn = trn.merge(feat, how='left', on='sid')
    tst = tst.merge(feat, how='left', on='sid')
    
    trn.drop(['sid', 'pid', 'click_mode'], axis=1, inplace=True)
    tst.drop(['sid', 'pid', 'click_mode'], axis=1, inplace=True)

    return trn, y, tst, sub

def get_train_test_features3():
    config.set_feature_name('f3')
    if os.path.exists(config.train_feature_file) and os.path.exists(config.test_feature_file):
        logger.info('loading the training and test features from files.')
        trn = pd.read_csv(config.train_feature_file)
        tst = pd.read_csv(config.test_feature_file)
    else:
        df = merge_raw_data()
        logger.info('generating feature f3.')
        trn, tst = generate_f3(df)

        logger.info('saving the training and test f3 features.')
        trn.to_csv(config.train_feature_file, index=False)
        tst.to_csv(config.test_feature_file, index=False)

    y = trn['click_mode'].values
    sub = tst[['sid']].copy()

    trn.drop(['sid', 'pid', 'click_mode'], axis=1, inplace=True)
    tst.drop(['sid', 'pid', 'click_mode'], axis=1, inplace=True)

    return trn, y, tst, sub

def get_train_test_features4():
    config.set_feature_name('f4')
    if os.path.exists(config.train_feature_file) and os.path.exists(config.test_feature_file):
        logger.info('loading the training and test features from files.')
        trn = pd.read_csv(config.train_feature_file)
        tst = pd.read_csv(config.test_feature_file)
    
    y = trn['click_mode'].values
    sub = tst[['sid']].copy()

    trn.drop(['sid', 'pid', 'click_mode'], axis=1, inplace=True)
    tst.drop(['sid', 'pid', 'click_mode'], axis=1, inplace=True)

    return trn, y, tst, sub

def get_train_test_features0():
    config.set_feature_name('f0')
    if os.path.exists(config.train_feature_file) and os.path.exists(config.test_feature_file):
        logger.info('loading the training and test features from files.')
        trn = pd.read_csv(config.train_feature_file)
        tst = pd.read_csv(config.test_feature_file)
    
    y = trn['click_mode'].values
    sub = tst[['sid']].copy()

    feat = pd.read_csv('/home/ubuntu/projects/kddcup2019track1/build/feature/od_coord_feature.csv')
    trn = trn.merge(feat, how='left', on='sid')
    tst = tst.merge(feat, how='left', on='sid')

    feat = pd.read_csv('/home/ubuntu/projects/kddcup2019track1/input/data_set_phase1/var_dist_time.csv')
    trn = trn.merge(feat, how='left', on='sid')
    tst = tst.merge(feat, how='left', on='sid')

    feat = pd.read_csv('/home/ubuntu/projects/kddcup2019track1/input/data_set_phase1/var_dist_min.csv')
    trn = trn.merge(feat, how='left', on='sid')
    tst = tst.merge(feat, how='left', on='sid')

    trn.drop(['sid', 'click_mode'], axis=1, inplace=True)
    tst.drop(['sid', 'click_mode'], axis=1, inplace=True)

    return trn, y, tst, sub


if __name__ == "__main__":
    df = merge_raw_data()

    if not os.path.exists(config.plan_file):
        logger.info('extracting plans from JSON objects.')
        plans = extract_plans(df)
        plans.to_csv(config.plan_file, index=False)

    if not os.path.exists(config.pid_feature_file):
        logger.info('generating pid features.')
        feat = generate_pid_features(df)
        feat.to_csv(config.pid_feature_file, index=False)

    if not os.path.exists(config.od_feature_file):
        logger.info('generating od features.')
        feat = generate_od_features(df)
        feat.to_csv(config.od_feature_file, index=False)

    if not os.path.exists(config.od_cluster_feature_file):
        logger.info('generating od cluster features.')
        feat = generate_od_cluster_features(df)
        feat.to_csv(config.od_cluster_feature_file, index=False)
