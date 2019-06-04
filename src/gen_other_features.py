
# coding: utf-8

# In[10]:
import numpy as np
import pandas as pd
from config import logger, config
from itertools import combinations


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))



# In[11]:

print('[+] loading data')
train = pd.read_csv(config.train_query_file)
test = pd.read_csv(config.test_query_file)

plans = pd.read_csv(config.plan_file)

# In[12]:

print('[+] generating features')
train['o_lat'] = train['o'].map(lambda x: float(x.split(',')[0]))
train['o_lon'] = train['o'].map(lambda x: float(x.split(',')[1]))

train['d_lat'] = train['d'].map(lambda x: float(x.split(',')[0]))
train['d_lon'] = train['d'].map(lambda x: float(x.split(',')[1]))

test['o_lat'] = test['o'].map(lambda x: float(x.split(',')[0]))
test['o_lon'] = test['o'].map(lambda x: float(x.split(',')[1]))

test['d_lat'] = test['d'].map(lambda x: float(x.split(',')[0]))
test['d_lon'] = test['d'].map(lambda x: float(x.split(',')[1]))

feat = pd.concat((train[['sid','o_lat', 'o_lon', 'd_lat', 'd_lon']], test[['sid','o_lat', 'o_lon', 'd_lat', 'd_lon']]))

print('[+] lon lat distance features')
feat['od_haversine_dist'] = feat[['o_lat', 'o_lon', 'd_lat', 'd_lon']].apply(lambda x: haversine_array(x[0], x[1], x[2], x[3]), axis=1)
feat['od_manhattan_dist'] = feat[['o_lat', 'o_lon', 'd_lat', 'd_lon']].apply(lambda x: dummy_manhattan_distance(x[0], x[1], x[2], x[3]), axis=1)
feat['od_bearing'] = feat[['o_lat', 'o_lon', 'd_lat', 'd_lon']].apply(lambda x: bearing_array(x[0], x[1], x[2], x[3]), axis=1)

print('[+] lon lat cluster features')
coords = np.vstack((train[['o_lat', 'o_lon']].values,
                    train[['d_lat', 'd_lon']].values,
                    test[['o_lat', 'o_lon']].values,
                    test[['d_lat', 'd_lon']].values))


from sklearn.cluster import MiniBatchKMeans

sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=50, batch_size=10000).fit(coords[sample_ind])
 
feat['o_coord_cluster'] = kmeans.predict(feat[['o_lat', 'o_lon']])
feat['d_coord_cluster'] = kmeans.predict(feat[['d_lat', 'd_lon']])

print('[+] lon lat PCA features')
from sklearn.decomposition import PCA
pca = PCA().fit(coords)
feat['o_coord_pca0'] = pca.transform(feat[['o_lat', 'o_lon']])[:, 0]
feat['o_coord_pca1'] = pca.transform(feat[['o_lat', 'o_lon']])[:, 1]
feat['d_coord_pca0'] = pca.transform(feat[['d_lat', 'd_lon']])[:, 0]
feat['d_coord_pca1'] = pca.transform(feat[['d_lat', 'd_lon']])[:, 1]

print('[+] agg features')
t = pd.crosstab(index=plans['sid'], columns=plans['transport_mode'], values=plans['distance'], aggfunc=np.mean)
for x,y in combinations(range(1,12),2):
    t['dist_mode_%d/%d' % (x,y)] = t[x] / t[y]

t.drop(list(range(1, 12)), axis=1, inplace=True)
feat = feat.merge(t.reset_index(), how='left', on='sid')

t = pd.crosstab(index=plans['sid'], columns=plans['transport_mode'], values=plans['price'], aggfunc=np.mean)
cols = t.columns.tolist()
for x,y in combinations(cols,2):
    t['price_mode_%d/%d' % (x,y)] = t[x] / t[y]

t.drop(cols, axis=1, inplace=True)
feat = feat.merge(t.reset_index(), how='left', on='sid')

t = pd.crosstab(index=plans['sid'], columns=plans['transport_mode'], values=plans['eta'], aggfunc=np.mean)
cols = t.columns.tolist()
for x,y in combinations(cols,2):
    t['eta_mode_%d/%d' % (x,y)] = t[x] / t[y]

t.drop(cols, axis=1, inplace=True)
feat = feat.merge(t.reset_index(), how='left', on='sid')

print('[+] saving features: %s' % config.od_coord_feature_file)
feat.drop(['o_lat', 'o_lon', 'd_lat', 'd_lon'], axis=1).to_csv(config.od_coord_feature_file, index=False)


# In[14]:

data = pd.concat((train, test))


# In[15]:

data['pid'] = data['pid'].fillna(-1).astype(int)


# In[16]:

data['o1'] = data['o'].map(lambda x: 100*float(x.split(',')[0])).astype(int)
data['o2'] = data['o'].map(lambda x: 100*float(x.split(',')[1])).astype(int)

data['d1'] = data['d'].map(lambda x: 100*float(x.split(',')[0])).astype(int)
data['d2'] = data['d'].map(lambda x: 100*float(x.split(',')[1])).astype(int)
data.head()


# In[17]:

data['req_time'] = pd.to_datetime(data['req_time'])
data['hour'] = data['req_time'].map(lambda x: x.hour)
data['day'] = data['req_time'].map(lambda x: x.day)
data['weekday'] = data['req_time'].map(lambda x: x.weekday())


t = plans.merge(data[['sid','o','d']], how='left', on='sid')
feat = t[['o','d']].groupby(['o','d']).head(1)
cnt = pd.crosstab(index=t['o'], columns=t['transport_mode'])
cnt.columns = pd.Index(['o_mode_{}_cnt'.format(x) for x in cnt.columns])
cnt.reset_index()
feat = feat.merge(cnt, how='left', on='o')

cnt = pd.crosstab(index=t['d'], columns=t['transport_mode'])
cnt.columns = pd.Index(['d_mode_{}_cnt'.format(x) for x in cnt.columns])
cnt.reset_index()
feat = feat.merge(cnt, how='left', on='d')

cnt = pd.crosstab(index=[t['o'], t['d']], columns=t['transport_mode'])
cnt.columns = pd.Index(['od_mode_{}_cnt'.format(x) for x in cnt.columns])
cnt.reset_index()
feat = feat.merge(cnt, how='left', on=['o','d'])

feat.to_csv(config.od_mode_cnt_feature_file, index=False)


feat = data.groupby(['hour','weekday'])['o'].nunique().reset_index()

feat = feat.merge(data.groupby(['hour','weekday'])['d'].nunique().reset_index(), how='left', on=['hour','weekday'])
feat = feat.merge(data.groupby(['hour','weekday'])['pid'].nunique().reset_index(), how='left', on=['hour','weekday'])
feat.rename(columns={'o': 'weekday_hour_o_cnt', 'd': 'weekday_hour_d_cnt', 'pid': 'weekday_hour_pid_cnt'}, inplace=True)
feat.to_csv(config.weekday_hour_feature_file, index=False)



t = plans.merge(data[['sid']], how='left', on='sid')
first_plan = t.groupby('sid').head(1)
first_plan.to_csv(config.od_plan_agg_feature_file, index=False)


feat = pd.crosstab(index=[plans['sid']], columns=plans['planid'], values=plans['transport_mode'], aggfunc=np.mean)
feat.columns = ['plan_code_%d' % c for c in feat.columns]
feat.to_csv(config.mode_feature_file)


'''
# In[19]:

data = data.merge(plans, how='right', on='sid')

# In[20]:

data.drop('req_time', axis=1, inplace=True)


# In[22]:

t = data.groupby(['o','d'])['transport_mode'].nunique()


# In[24]:

t = data.groupby(['o','d','hour'])['transport_mode'].nunique()


# In[27]:

t = data.groupby(['o','d','hour','weekday'])['transport_mode'].nunique()


# In[30]:

t = data.groupby(['o','d','hour','weekday','transport_mode'])['distance'].std()


# In[31]:

t.head()


# In[32]:

t = pd.crosstab(index=[data['o'],data['d'],data['hour'],data['weekday']], columns=data['transport_mode'], values=data['distance'], aggfunc=np.std)


# In[33]:

t.head()


# In[34]:

feat = data[['sid','o','d','hour','weekday']].groupby('sid').head(1)
t = data.groupby(['o','d'])['transport_mode'].nunique().to_frame(name='od_transport_mode_nunique').reset_index()
feat = feat.merge(t, how='left', on=['o', 'd'])
feat.head()


# In[69]:

t = data.groupby(['o','d','hour'])['transport_mode'].nunique().to_frame(name='od_hour_transport_mode_nunique').reset_index()
feat = feat.merge(t, how='left', on=['o', 'd', 'hour'])
feat.head()


# In[70]:

t = data.groupby(['o','d','hour','weekday'])['transport_mode'].nunique().to_frame(name='od_hour_week_transport_mode_nunique').reset_index()
feat = feat.merge(t, how='left', on=['o', 'd', 'hour','weekday'])
feat.head()


# In[71]:

t = data.groupby(['o','d','hour'])['distance'].std().to_frame(name='od_hour_dist_std').reset_index()
feat = feat.merge(t, how='left', on=['o', 'd', 'hour'])
feat.head()


# In[72]:

t = data.groupby(['o','d','hour','weekday'])['distance'].std().to_frame(name='od_hour_week_dist_std').reset_index()
feat = feat.merge(t, how='left', on=['o', 'd', 'hour','weekday'])
feat.head()


# In[75]:

t = pd.crosstab(index=[data['o'],data['d'],data['hour'],data['weekday']], columns=data['transport_mode'], values=data['distance'], aggfunc=np.std)
t.columns = ['od_hour_weekday_mode_%d_dist_std' % c for c in t.columns]
t = t.reset_index()
feat = feat.merge(t, how='left', on=['o', 'd', 'hour','weekday'])
feat.head()


# In[76]:

t = pd.crosstab(index=[data['o'],data['d'],data['hour']], columns=data['transport_mode'], values=data['distance'], aggfunc=np.std)
t.columns = ['od_hour_mode_%d_dist_std' % c for c in t.columns]
t = t.reset_index()
feat = feat.merge(t, how='left', on=['o', 'd', 'hour'])
feat.head()


# In[77]:

feat.drop(['o','d','hour','weekday'], axis=1).to_csv('../build/feature/od_mode_stats.csv', index=False)


# In[78]:

data.head()


# In[113]:

t = data.groupby(['o','d'])['distance'].mean().to_frame(name='od_dist_mean').reset_index()


# In[114]:

t = data[['sid','o','d','distance','planid']].merge(t, how='left', on=['o','d'])


# In[115]:

t['dist-d-od_dist_mean'] = t['distance'] / t['od_dist_mean']


# In[116]:

t = t[t['planid'] <= 3]


# In[117]:

feat = pd.crosstab(index=t['sid'], columns=t['planid'], values=t['dist-d-od_dist_mean'], aggfunc=np.mean)


# In[118]:

feat.columns = ['dist-d-od_dist_mean_plan_%d' % c for c in feat.columns]


# In[119]:

t1 = t.groupby('sid')['dist-d-od_dist_mean'].agg([np.min, np.max, np.mean, np.std])
t1.columns = ['dist-d-od_dist_mean_min', 'dist-d-od_dist_max', 'dist-d-od_dist_mean_mean','dist-d-od_dist_mean_std']


# In[120]:

feat = feat.reset_index().merge(t1.reset_index(), how='left', on='sid')


# In[121]:

feat.to_csv('../build/feature/od_mode_stats.csv', index=False)


# In[ ]:

feat = data.groupby(['sid','o', 'd']).head(1)

feat['o_nearest_dis'] = np.nan
feat['d_nearest_dis'] = np.nan


co_o = feat[['o']]
co_d = feat[['d']]

co_o.columns = ['co']
co_d.columns = ['co']

all_co = pd.concat([co_d, co_o])['co'].unique()

for co in tqdm(all_co):
    lg, la = co.split(',')
    min_dis = (abs(subwayinfo['station_longitude']-float(lg)) +
               abs(subwayinfo['station_latitude']-float(la))).min()
    feat.loc[(data['o'] == co), 'o_nearest_dis'] = min_dis
    feat.loc[(data['d'] == co), 'd_nearest_dis'] = min_dis
    


# In[129]:

feat = data[['sid','o','d']].groupby('sid').head(1)
t = data[['o','d']].groupby(['o','d']).head(1)

feat['o_lng'] = feat['o'].apply(lambda x: float(x.split(',')[0]))
feat['o_lat'] = feat['o'].apply(lambda x: float(x.split(',')[1]))

feat['d_lng'] = feat['d'].apply(lambda x: float(x.split(',')[0]))
feat['d_lat'] = feat['d'].apply(lambda x: float(x.split(',')[1]))


# In[125]:

o_co = t[['o']]
d_co = t[['d']]

o_co.columns = ['co']
d_co.columns = ['co']

all_co = pd.concat([d_co, o_co]).drop_duplicates()
all_co.shape


# In[126]:

all_co['lng'] = all_co['co'].apply(lambda x: float(x.split(',')[0]))
all_co['lat'] = all_co['co'].apply(lambda x: float(x.split(',')[1]))


# In[127]:

lng_mean = all_co['lng'].mean()
lat_mean = all_co['lat'].mean()

lng_mode = all_co['lng'].mode()[0]
lat_mode = all_co['lat'].mode()[0]


# In[130]:

feat['o_main_centroid_mean_dis'] = abs(
    feat['o_lng']-lng_mean)+abs(feat['o_lat']-lat_mean)
feat['d_main_centroid_mean_dis'] = abs(
    feat['d_lng']-lng_mean)+abs(feat['d_lat']-lat_mean)

feat['o_main_centroid_mode_dis'] = abs(
    feat['o_lng']-lng_mean)+abs(feat['o_lat']-lat_mean)
feat['d_main_centroid_mode_dis'] = abs(
    feat['d_lng']-lng_mode)+abs(feat['d_lat']-lat_mode)


# In[133]:

feat['od_manhattan_distance'] = abs(
        feat['o_lng']-feat['d_lng'])+abs(feat['o_lat']-feat['d_lat'])


# In[134]:

feat.head()


# In[135]:

feat.drop(['o','d','o_lng','o_lat','d_lng','d_lat'], axis=1).to_csv('../build/feature/od_mode_stats.csv', index=False)


# In[140]:

# t = data.groupby('o', as_index=False)['hour'].agg({'o_hour_min': np.min, 'o_hour_max': np.max})
# feat = feat.merge(t, how='left', on='o')


# In[146]:

# t = data.groupby('d', as_index=False)['hour'].agg({'d_hour_min': np.min, 'd_hour_max': np.max})
# feat = feat.merge(t, how='left', on='d')


# In[153]:

t = data.groupby('o')['hour'].agg({'o_hour_mode1': lambda x: x.value_counts().index[0], 
                                   'o_hour_mode2': lambda x: x.value_counts().index[1] if x.nunique() > 1 else x.value_counts().index[0]}).reset_index()

t['o_hour_mode1-m-o_hour_mode2'] = t['o_hour_mode1'] - t['o_hour_mode2']
feat = feat.merge(t, how='left', on='o')


# In[154]:

t = data.groupby('d')['hour'].agg({'d_hour_mode1': lambda x: x.value_counts().index[0], 
                                   'd_hour_mode2': lambda x: x.value_counts().index[1] if x.nunique() > 1 else x.value_counts().index[0]}).reset_index()

t['d_hour_mode1-m-d_hour_mode2'] = t['d_hour_mode1'] - t['d_hour_mode2']
feat = feat.merge(t, how='left', on='d')


# In[155]:

feat.head()


# In[156]:

feat.drop(['o','d','o_lng','o_lat','d_lng','d_lat'], axis=1).to_csv('../build/feature/od_mode_stats.csv', index=False)


# In[171]:

weather = pd.read_csv('../input/data_set_phase1/weather.csv')


# In[172]:

weather.rename(columns={'date': 'req_time'}, inplace=True)


# In[181]:

weather['weather'] = (weather['weather'].map(
        {'q': 0, 'dy': 1, 'dyq': 2, 'qdy': 3, 'xq': 4, 'xydy': 5})).astype(int)


# In[182]:

feat = pd.read_csv('../input/data_set_phase1/train_queries.csv', usecols=['sid', 'req_time'])
feat = pd.concat((feat, pd.read_csv('../input/data_set_phase1/test_queries.csv', usecols=['sid', 'req_time'])))


# In[183]:

feat['req_time'] = feat['req_time'].map(lambda x: x.split(' ')[0])
feat['req_time'] = feat['req_time'].map(lambda x: '-'.join(x.split('-')[1:]))


# In[184]:

feat = feat.merge(weather, how='left', on='req_time')


# In[185]:

feat.drop('req_time', axis=1).to_csv('../build/feature/weather_features.csv', index=False)


# In[186]:

feat.head()


# In[176]:

location = pd.read_csv('../input/data_set_phase1/address_info_clean.csv')


# In[177]:

location.head()


# In[194]:

feat = pd.read_csv(config.train_query_file, usecols=['sid', 'req_time','o','d','pid'], parse_dates=['req_time'])
feat = pd.concat((feat, pd.read_csv(config.test_query_file, usecols=['sid', 'req_time', 'o', 'd', 'pid'], parse_dates=['req_time'])))


# In[195]:

feat['req_date'] = feat['req_time'].map(lambda x: x.date())
feat.head()


# In[196]:

feat['req_hour'] = feat['req_time'].map(lambda x: x.replace(minute=0, second=0))
feat.head()


# In[197]:

feat['req_minute'] = feat['req_time'].map(lambda x: x.minute+x.hour*60)


# In[198]:

feat.head()


# In[199]:

feat.drop('req_time', axis=1, inplace=True)


# In[200]:

feat = plans.merge(feat, how='left', on='sid')


# In[201]:

feat.head()


# In[202]:

t = feat.copy()


# In[203]:

#t = t[~pd.isnull(t['pid'])]


# In[204]:

feat = t.groupby('sid').head(1)[['sid','req_date','req_hour','req_minute','pid','o','d']]


# In[205]:

dly_agg = t.groupby('req_date', as_index=False)['distance'].agg({'daily_dist_min': np.min,
                                                                 'daily_dist_max': np.max, 
                                                                 'daily_dist_mean': np.mean, 
                                                                 'daily_dist_std': np.std})


# In[206]:

hly_agg = t.groupby('req_hour', as_index=False)['distance'].agg({'hourly_dist_min': np.min,
                                                                 'hourly_dist_max': np.max,
                                                                 'hourly_dist_mean': np.mean, 
                                                                 'hourly_dist_std': np.std})


# In[207]:

feat = feat.merge(dly_agg, how='left', on='req_date')
feat = feat.merge(hly_agg, how='left', on='req_hour')


# In[208]:

dly_agg = t.groupby('req_date', as_index=False)['eta'].agg({'daily_eta_min': np.min,
                                                            'daily_eta_max': np.max,
                                                            'daily_eta_mean': np.mean, 
                                                            'daily_eta_std': np.std})

hly_agg = t.groupby('req_hour', as_index=False)['eta'].agg({'hourly_eta_max': np.max, 
                                                            'hourly_eta_min': np.min, 
                                                            'hourly_eta_mean': np.mean, 
                                                            'hourly_eta_std': np.std})

feat = feat.merge(dly_agg, how='left', on='req_date')
feat = feat.merge(hly_agg, how='left', on='req_hour')


# In[209]:

dly_agg = t.groupby('req_date', as_index=False)['price'].agg({'daily_price_min': np.min, 
                                                              'daily_price_max': np.max, 
                                                              'daily_price_mean': np.mean, 
                                                              'daily_price_std': np.std})

hly_agg = t.groupby('req_hour', as_index=False)['price'].agg({'hourly_price_min': np.min, 
                                                              'hourly_price_max': np.max, 
                                                            'hourly_price_mean': np.mean, 
                                                            'hourly_price_std': np.std})

feat = feat.merge(dly_agg, how='left', on='req_date')
feat = feat.merge(hly_agg, how='left', on='req_hour')


# In[210]:

hly_agg = t.groupby('req_hour', as_index=False)['transport_mode'].agg({'hourly_mode_nunique': "nunique", 
                                                                       'hourly_mode_mode': lambda x: x.value_counts().index[0],
                                                                      })


# In[211]:

feat = feat.merge(hly_agg, how='left', on='req_hour')


# In[212]:

feat.drop(['req_date', 'req_hour', 'pid', 'o', 'd'], axis=1).to_csv('../build/feature/daily_plan_features.csv', index=False)


# In[213]:

feat.head()


# In[129]:

t['pid'] = t['pid'].fillna(-1)


# In[130]:

dly_agg = t.groupby(['req_date','pid'], as_index=False)['eta'].agg({'daily_pid_eta_max': np.max, 
                                                            'daily_pid_eta_mean': np.mean, 
                                                            'daily_pid_eta_std': np.std})

hly_agg = t.groupby(['req_hour','pid'], as_index=False)['eta'].agg({'hourly_pid_eta_max': np.max, 
                                                            'hourly_pid_eta_mean': np.mean, 
                                                            'hourly_pid_eta_std': np.std})

feat = feat.merge(dly_agg, how='left', on=['req_date','pid'])
feat = feat.merge(hly_agg, how='left', on=['req_hour','pid'])


# In[131]:

dly_agg = t.groupby(['req_date','pid'], as_index=False)['distance'].agg({'daily_pid_dist_max': np.max, 
                                                            'daily_pid_dist_mean': np.mean, 
                                                            'daily_pid_dist_std': np.std})

hly_agg = t.groupby(['req_hour','pid'], as_index=False)['distance'].agg({'hourly_pid_dist_max': np.max, 
                                                            'hourly_pid_dist_mean': np.mean, 
                                                            'hourly_pid_dist_std': np.std})

feat = feat.merge(dly_agg, how='left', on=['req_date','pid'])
feat = feat.merge(hly_agg, how='left', on=['req_hour','pid'])


# In[132]:

dly_agg = t.groupby(['req_date','pid'], as_index=False)['price'].agg({'daily_pid_price_max': np.max, 
                                                            'daily_pid_price_mean': np.mean, 
                                                            'daily_pid_price_std': np.std})

hly_agg = t.groupby(['req_hour','pid'], as_index=False)['price'].agg({'hourly_pid_price_max': np.max, 
                                                            'hourly_pid_price_mean': np.mean, 
                                                            'hourly_pid_price_std': np.std})

feat = feat.merge(dly_agg, how='left', on=['req_date','pid'])
feat = feat.merge(hly_agg, how='left', on=['req_hour','pid'])


# In[133]:

feat.drop(['req_date', 'req_hour', 'pid', 'o', 'd'], axis=1).to_csv('../build/feature/daily_plan_features.csv', index=False)


# In[134]:

dly_agg = t.groupby(['req_date','o'], as_index=False)['eta'].agg({'daily_o_eta_max': np.max, 
                                                            'daily_o_eta_mean': np.mean, 
                                                            'daily_o_eta_std': np.std})

hly_agg = t.groupby(['req_hour','o'], as_index=False)['eta'].agg({'hourly_o_eta_max': np.max, 
                                                            'hourly_o_eta_mean': np.mean, 
                                                            'hourly_o_eta_std': np.std})

feat = feat.merge(dly_agg, how='left', on=['req_date','o'])
feat = feat.merge(hly_agg, how='left', on=['req_hour','o'])


# In[ ]:




# In[135]:

dly_agg = t.groupby(['req_date','o'], as_index=False)['distance'].agg({'daily_o_dist_max': np.max, 
                                                            'daily_o_dist_mean': np.mean, 
                                                            'daily_o_dist_std': np.std})

hly_agg = t.groupby(['req_hour','o'], as_index=False)['distance'].agg({'hourly_o_dist_max': np.max, 
                                                            'hourly_o_dist_mean': np.mean, 
                                                            'hourly_o_dist_std': np.std})

feat = feat.merge(dly_agg, how='left', on=['req_date','o'])
feat = feat.merge(hly_agg, how='left', on=['req_hour','o'])


# In[136]:

dly_agg = t.groupby(['req_date','o'], as_index=False)['price'].agg({'daily_o_price_max': np.max, 
                                                            'daily_o_price_mean': np.mean, 
                                                            'daily_o_price_std': np.std})

hly_agg = t.groupby(['req_hour','o'], as_index=False)['price'].agg({'hourly_o_price_max': np.max, 
                                                            'hourly_o_price_mean': np.mean, 
                                                            'hourly_o_price_std': np.std})

feat = feat.merge(dly_agg, how='left', on=['req_date','o'])
feat = feat.merge(hly_agg, how='left', on=['req_hour','o'])


# In[137]:

dly_agg = t.groupby(['req_date','d'], as_index=False)['eta'].agg({'daily_d_eta_max': np.max, 
                                                            'daily_d_eta_mean': np.mean, 
                                                            'daily_d_eta_std': np.std})

hly_agg = t.groupby(['req_hour','d'], as_index=False)['eta'].agg({'hourly_d_eta_max': np.max, 
                                                            'hourly_d_eta_mean': np.mean, 
                                                            'hourly_d_eta_std': np.std})

feat = feat.merge(dly_agg, how='left', on=['req_date','d'])
feat = feat.merge(hly_agg, how='left', on=['req_hour','d'])

dly_agg = t.groupby(['req_date','d'], as_index=False)['distance'].agg({'daily_d_dist_max': np.max, 
                                                            'daily_d_dist_mean': np.mean, 
                                                            'daily_d_dist_std': np.std})

hly_agg = t.groupby(['req_hour','d'], as_index=False)['distance'].agg({'hourly_d_dist_max': np.max, 
                                                            'hourly_d_dist_mean': np.mean, 
                                                            'hourly_d_dist_std': np.std})

feat = feat.merge(dly_agg, how='left', on=['req_date','d'])
feat = feat.merge(hly_agg, how='left', on=['req_hour','d'])

dly_agg = t.groupby(['req_date','d'], as_index=False)['price'].agg({'daily_d_price_max': np.max, 
                                                            'daily_d_price_mean': np.mean, 
                                                            'daily_d_price_std': np.std})

hly_agg = t.groupby(['req_hour','d'], as_index=False)['price'].agg({'hourly_d_price_max': np.max, 
                                                            'hourly_d_price_mean': np.mean, 
                                                            'hourly_d_price_std': np.std})

feat = feat.merge(dly_agg, how='left', on=['req_date','d'])
feat = feat.merge(hly_agg, how='left', on=['req_hour','d'])


# In[138]:

feat.drop(['req_date', 'req_hour', 'pid', 'o', 'd'], axis=1).to_csv('../build/feature/daily_plan_features.csv', index=False)


# In[36]:

data.head()


# In[37]:

t = data.drop_duplicates(subset='sid')


# In[41]:

t.head()


# In[55]:

click = pd.read_csv('../input/data_set_phase1/train_clicks.csv')


# In[56]:

click = click.merge(t[['sid','o','d','pid','hour','transport_mode']], how='left', on='sid')


# In[57]:

click.head()


# In[58]:

click = click[click['click_mode']!=-1]


# In[59]:

cnt = click.groupby(['o','d','pid'])['click_mode'].nunique()


# In[60]:

cnt.hist()


# In[61]:

cnt1 = click.groupby(['o','d','pid'])['transport_mode'].nunique()


# In[62]:

cnt1.hist()


# In[67]:

click = click[click['pid']!=-1]


# In[68]:

cnt = click.groupby(['o','d','pid'])['click_mode'].nunique()
cnt.hist()


# In[69]:

cnt = click.groupby(['o','d','pid'])['transport_mode'].nunique()

# In[76]:

cnt = click.groupby(['o','d','pid'])['transport_mode'].count()


# In[102]:

click = pd.read_csv('../input/data_set_phase1/train_clicks.csv')
click = t[['sid','o','d','pid','hour','transport_mode']].merge(click, how='left', on='sid')
click = click[click['pid']!=-1]


# In[103]:

cnt = click.groupby(['o','d','pid'])['transport_mode'].count().to_frame(name='cnt')
n_cnt = click.groupby(['o','d','pid'])['transport_mode'].nunique().to_frame(name='nunique')


# In[104]:

cnt = cnt.merge(n_cnt, left_index=True, right_index=True)


# In[105]:

cnt['count_ratio'] = cnt['nunique'] / cnt['cnt']


# In[106]:

cnt = cnt.reset_index()
cnt = t[['sid','o','d','pid']].merge(cnt, how='left', on=['o','d','pid'])


# In[107]:

cnt[['sid','count_ratio']].to_csv('../build/feature/od_pid_count.csv', index=False)


# In[216]:

mean_plan = data.groupby(['o','d'])[['distance','eta','price']].mean().reset_index()
mean_plan.head()


# In[219]:

feat = data.groupby(['sid','o','d'])[['distance','eta','price']].mean().reset_index()


# In[221]:

feat = feat.merge(mean_plan, how='left', on=['o','d'])


# In[223]:

feat['dist_ratio'] = feat['distance_x'] / feat['distance_y']
feat['eta_ratio'] = feat['eta_x'] / feat['eta_y']
feat['price_ratio'] = feat['price_x'] / feat['price_y']


# In[227]:

feat[['sid', 'dist_ratio', 'eta_ratio', 'price_ratio']].to_csv('../build/feature/plan_ratio.csv', index=False)

# In[228]:

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


# In[229]:

train = pd.read_csv(config.train_query_file, usecols=['sid','o','d'])
test = pd.read_csv(config.test_query_file, usecols=['sid','o','d'])


# In[231]:

train['o_lat'] = train['o'].map(lambda x: float(x.split(',')[0]))
train['o_lon'] = train['o'].map(lambda x: float(x.split(',')[1]))

train['d_lat'] = train['d'].map(lambda x: float(x.split(',')[0]))
train['d_lon'] = train['d'].map(lambda x: float(x.split(',')[1]))

test['o_lat'] = test['o'].map(lambda x: float(x.split(',')[0]))
test['o_lon'] = test['o'].map(lambda x: float(x.split(',')[1]))

test['d_lat'] = test['d'].map(lambda x: float(x.split(',')[0]))
test['d_lon'] = test['d'].map(lambda x: float(x.split(',')[1]))


# In[232]:

feat = pd.concat((train[['sid','o_lat', 'o_lon', 'd_lat', 'd_lon']], test[['sid','o_lat', 'o_lon', 'd_lat', 'd_lon']]))


# In[238]:

feat['od_haversine_dist'] = feat[['o_lat', 'o_lon', 'd_lat', 'd_lon']].apply(lambda x: haversine_array(x[0], x[1], x[2], x[3]), axis=1)
feat['od_manhattan_dist'] = feat[['o_lat', 'o_lon', 'd_lat', 'd_lon']].apply(lambda x: dummy_manhattan_distance(x[0], x[1], x[2], x[3]), axis=1)
feat['od_bearing'] = feat[['o_lat', 'o_lon', 'd_lat', 'd_lon']].apply(lambda x: bearing_array(x[0], x[1], x[2], x[3]), axis=1)


# In[233]:

coords = np.vstack((train[['o_lat', 'o_lon']].values,
                    train[['d_lat', 'd_lon']].values,
                    test[['o_lat', 'o_lon']].values,
                    test[['d_lat', 'd_lon']].values))


# In[240]:

from sklearn.cluster import MiniBatchKMeans

sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
 
feat['o_coord_cluster'] = kmeans.predict(feat[['o_lat', 'o_lon']])
feat['d_coord_cluster'] = kmeans.predict(feat[['d_lat', 'd_lon']])


# In[243]:

from sklearn.decomposition import PCA
pca = PCA().fit(coords)
feat['o_coord_pca0'] = pca.transform(feat[['o_lat', 'o_lon']])[:, 0]
feat['o_coord_pca1'] = pca.transform(feat[['o_lat', 'o_lon']])[:, 1]
feat['d_coord_pca0'] = pca.transform(feat[['d_lat', 'd_lon']])[:, 0]
feat['d_coord_pca1'] = pca.transform(feat[['d_lat', 'd_lon']])[:, 1]



# In[245]:

feat.drop(['o_lat', 'o_lon', 'd_lat', 'd_lon'], axis=1).to_csv('../build/feature/od_coord_feature.csv', index=False)


# In[246]:

feat.drop(['o_lat', 'o_lon', 'd_lat', 'd_lon'], axis=1).head()


# In[247]:

plans.head()


# In[257]:

from itertools import combinations


# In[262]:

t = pd.crosstab(index=plans['sid'], columns=plans['planid'], values=plans['distance'], aggfunc=np.mean)
for x,y in combinations(range(7),2):
    t['dist_plan_%d/%d' % (x,y)] = t[x] / t[y]
    t['dist_plan_%d-%d' % (x,y)] = t[x] - t[y]
    
t.drop(list(range(7)), axis=1, inplace=True)


# In[264]:

feat = feat.merge(t.reset_index(), how='left', on='sid')


# In[266]:

t = pd.crosstab(index=plans['sid'], columns=plans['planid'], values=plans['price'], aggfunc=np.mean)
for x,y in combinations(range(7),2):
    t['price_plan_%d/%d' % (x,y)] = t[x] / t[y]
    t['price_plan_%d-%d' % (x,y)] = t[x] - t[y]
    
t.drop(list(range(7)), axis=1, inplace=True)


# In[267]:

feat = feat.merge(t.reset_index(), how='left', on='sid')
feat.head()


# In[268]:

t = pd.crosstab(index=plans['sid'], columns=plans['planid'], values=plans['eta'], aggfunc=np.mean)
for x,y in combinations(range(7),2):
    t['eta_plan_%d/%d' % (x,y)] = t[x] / t[y]
    t['eta_plan_%d-%d' % (x,y)] = t[x] - t[y]
    
t.drop(list(range(7)), axis=1, inplace=True)


# In[269]:

feat.drop(['o_lat', 'o_lon', 'd_lat', 'd_lon'], axis=1).to_csv('../build/feature/od_coord_feature.csv', index=False)


# In[271]:

t = pd.crosstab(index=plans['sid'], columns=plans['transport_mode'], values=plans['distance'], aggfunc=np.mean)
for x,y in combinations(range(1,12),2):
    t['dist_mode_%d/%d' % (x,y)] = t[x] / t[y]
    t['dist_mode_%d-%d' % (x,y)] = t[x] - t[y]

t.drop(list(range(1, 12)), axis=1, inplace=True)


# In[272]:

feat = feat.merge(t.reset_index(), how='left', on='sid')


# In[275]:

t = pd.crosstab(index=plans['sid'], columns=plans['transport_mode'], values=plans['price'], aggfunc=np.mean)
cols = t.columns.tolist()
for x,y in combinations(cols,2):
    t['price_mode_%d/%d' % (x,y)] = t[x] / t[y]
    t['price_mode_%d-%d' % (x,y)] = t[x] - t[y]

t.drop(cols, axis=1, inplace=True)


# In[276]:

feat = feat.merge(t.reset_index(), how='left', on='sid')


# In[277]:

t = pd.crosstab(index=plans['sid'], columns=plans['transport_mode'], values=plans['eta'], aggfunc=np.mean)
cols = t.columns.tolist()
for x,y in combinations(cols,2):
    t['eta_mode_%d/%d' % (x,y)] = t[x] / t[y]
    t['eta_mode_%d-%d' % (x,y)] = t[x] - t[y]

t.drop(cols, axis=1, inplace=True)


# In[278]:

feat = feat.merge(t.reset_index(), how='left', on='sid')


# In[280]:

drop_cols = [x for x in feat.columns if x.find('-')>0]


# In[281]:

feat.drop(['o_lat', 'o_lon', 'd_lat', 'd_lon']+drop_cols, axis=1).to_csv('../build/feature/od_coord_feature.csv', index=False)


# In[ ]:
'''


