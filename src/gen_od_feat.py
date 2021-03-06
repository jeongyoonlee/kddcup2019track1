import pandas as pd
from config import logger, config
import numpy as np

def read_profile_data():
    profile_data = pd.read_csv(config.profile_file)
    profile_na = np.zeros(67)
    profile_na[0] = -1
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))
    profile_na.columns = profile_data.columns
    profile_data = profile_data.append(profile_na)
    return profile_data


def merge_raw_data():
    tr_queries = pd.read_csv(config.train_query_file)
    te_queries = pd.read_csv(config.test_query_file)
    tr_plans = pd.read_csv(config.train_plan_file)
    te_plans = pd.read_csv(config.test_plan_file)

    tr_click = pd.read_csv(config.train_click_file)

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


if __name__ == "__main__":
    data = merge_raw_data()
    data['req_time'] = data['req_time'].map(lambda x: pd.to_datetime(x))
    data['day'] = data['req_time'].map(lambda x: x.day)
    data['hour'] = data['req_time'].map(lambda x: x.hour)

    feat = data[['o','d']].groupby(['o','d']).head(1)
    feat = feat.merge(data.groupby('o')['day'].nunique().to_frame(name='o_nunique_day').reset_index(), how='left', on='o')
    feat = feat.merge(data.groupby('o')['hour'].nunique().to_frame(name='o_nunique_hour').reset_index(), how='left', on='o')
    feat = feat.merge(data.groupby('o')['pid'].nunique().to_frame(name='o_nunique_pid').reset_index(), how='left', on='o')
    feat = feat.merge(data.groupby('o')['click_mode'].nunique().to_frame(name='o_nunique_click').reset_index(), how='left', on='o')

    feat = feat.merge(data.groupby('d')['day'].nunique().to_frame(name='d_nunique_day').reset_index(), how='left', on='d')
    feat = feat.merge(data.groupby('d')['hour'].nunique().to_frame(name='d_nunique_hour').reset_index(), how='left', on='d')
    feat = feat.merge(data.groupby('d')['pid'].nunique().to_frame(name='d_nunique_pid').reset_index(), how='left', on='d')
    feat = feat.merge(data.groupby('d')['click_mode'].nunique().to_frame(name='d_nunique_click').reset_index(), how='left', on='d')
    
    feat = feat.merge(data.groupby(['o','d'])['day'].nunique().to_frame(name='od_nunique_day').reset_index(), how='left', on=['o','d'])
    feat = feat.merge(data.groupby(['o','d'])['hour'].nunique().to_frame(name='od_nunique_hour').reset_index(), how='left', on=['o','d'])
    feat = feat.merge(data.groupby(['o','d'])['pid'].nunique().to_frame(name='od_nunique_pid').reset_index(), how='left', on=['o','d'])
    feat = feat.merge(data.groupby(['o','d'])['click_mode'].nunique().to_frame(name='od_nunique_click').reset_index(), how='left', on=['o','d'])

    feat.to_csv(config.od_feature_file, index=False)
    
    
