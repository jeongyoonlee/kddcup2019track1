import pandas as pd
import networkx as nx
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
    G = nx.Graph()
    G.add_nodes_from(data['o'].unique().tolist())
    G.add_nodes_from(data['d'].unique().tolist())
    
    edges = data[['o','d']].apply(lambda x: (x[0],x[1]), axis=1).tolist()
    G.add_edges_from(edges)
    cluster = nx.clustering(G)
    
    cluster_df = pd.DataFrame([{'od': key, 'cluster': cluster[key]} for key in cluster.keys()])
    cluster_df.to_csv(config.od_cluster_feature_file, index=False)
    
    
