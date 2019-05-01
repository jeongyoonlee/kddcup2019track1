import pandas as pd
import json
import 

def get_plans(data):
    cur_plan_list = []
    for i, (sid, plan) in tqdm(enumerate(zip(data['sid'].values, data['plans'].values))):
        try:
            p = json.loads(plan)
            for x in p:
                x['sid'] = sid
            cur_plan_list.extend(p)            
        except:
            pass
        
    return pd.DataFrame(cur_plan_list)


if __name__ == "__main__":
    data = pd.concat((pd.read_csv('../input/train_plans.csv'),pd.read_csv('../input/test_plans.csv')))
    plans = get_plans(data)
    plans.to_csv('../data/plans.csv', index=False)