# KDD Cup 2019 Track 1

## Prepare data
* Create input folder and copy training and testing data into the folder
* Create folder submit
* Create folder feat

## Generate feature
* Run gen_pid_feat.py
* Run gen_od_feat.py
* Run gen_od_cluster_feat.py
* Run extract_plan.py to convert plans from json to csv. It will create plans.csv under input folder

## Train and generate submission
* Run train_model.py
