import pandas as pd
import csv

path = 'logs/log'
filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]

ml_list, taskid_list, runid_list, score_list = [], [], [], []
for i in range(6):
    # print("file ", i+1)
    with open(path+str(i+1), mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            ml_list.append(row['ml'])
            taskid_list.append(int(row['taskid']))
            runid_list.append(int(row['runid']))
            score_list.append(round(float(row['score']), 4))

dict_res = {
    'ml': ml_list,
    'taskid': taskid_list,
    'runid': runid_list,
    'score': score_list,
}
df_res = pd.DataFrame(dict_res)

df_nn_with_embedding = df_res[df_res['ml']=='nn_with_embedding']
df_nn = df_res[df_res['ml']=='nn']
df_random_forest = df_res[df_res['ml']=='random_forest']

mean_nn_with_embedding = df_nn_with_embedding.groupby(['taskid'])['score'].mean()
mean_nn = df_nn.groupby(['taskid'])['score'].mean()
mean_random_forest = df_random_forest.groupby(['taskid'])['score'].mean()

std_nn_with_embedding = df_nn_with_embedding.groupby(['taskid'])['score'].std()
std_nn = df_nn.groupby(['taskid'])['score'].std()
std_random_forest = df_random_forest.groupby(['taskid'])['score'].std()

dict_stats = {
    'taskid': filtered_tasks_ids,
    'nn_emb_mean': list(mean_nn_with_embedding.round(4)),
    'nn_mean': list(mean_nn.round(4)),
    'rf_mean': list(mean_random_forest.round(4)),
    'nn_emb_std': list(std_nn_with_embedding.round(4)),
    'nn_std': list(std_nn.round(4)),
    'rf_std': list(std_random_forest.round(4)),
}
df_stats = pd.DataFrame(dict_stats)
print(df_stats)
