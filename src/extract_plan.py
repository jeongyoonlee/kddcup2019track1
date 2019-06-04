import pandas as pd
import json
from tqdm import tqdm
from config import logger, config
 
def get_plans(data):
    cur_plan_list = []
    for i, (sid, plan) in tqdm(enumerate(zip(data['sid'].values, data['plans'].values))):
        try:
            p = json.loads(plan)
            for planid, x in enumerate(p):
                x['sid'] = sid
                x['planid'] = planid
            cur_plan_list.extend(p)            
        except:
            pass
        
    return pd.DataFrame(cur_plan_list)

if __name__ == "__main__":
    data = pd.concat((pd.read_csv(config.train_plan_file),pd.read_csv(config.test_plan_file)))
    plans = get_plans(data)
    plans.to_csv(config.plan_file, index=False)