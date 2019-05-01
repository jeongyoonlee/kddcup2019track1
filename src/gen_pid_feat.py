import pandas as pd

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


if __name__ == "__main__":
    data = merge_raw_data()
    data['req_time'] = data['req_time'].map(lambda x: pd.to_datetime(x))
    data['day'] = data['req_time'].map(lambda x: x.day)
    data['hour'] = data['req_time'].map(lambda x: x.hour)
    
    feat = data.groupby('pid')['hour'].nunique().to_frame(name='pid_nunique_hour').reset_index()
    feat = feat.merge(data.groupby('pid')['day'].nunique().to_frame(name='pid_nunique_day').reset_index(), how='left', on='pid')
    feat['nunique_hour-d-nunique_day'] = feat['pid_nunique_hour'] / feat['pid_nunique_day']
    feat = feat.merge(data.groupby('pid')['o'].nunique().to_frame(name='pid_nunique_o').reset_index(), how='left', on='pid')
    feat = feat.merge(data.groupby('pid')['d'].nunique().to_frame(name='pid_nunique_d').reset_index(), how='left', on='pid')
    feat['nunique_o-d-nunique_d'] = feat['pid_nunique_o'] / feat['pid_nunique_d']
    feat.to_csv('../feat/pid_feat.csv', index=False)
    
    
