import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

j = 0
filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]
strategy_list, taskid_list, runid_list, num_samples_list, score_r2_list, score_mse_list, score_mae_list, score_evc_list = [], [], [], [], [], [], [], []

blackbox = 'rf' # 'nn'
# related to task filtered_tasks_ids[0]
if blackbox == 'rf':
    path = 'logs/rf_local_fidelity_'+str(j)
else:
    path = 'logs/local_fidelity_'+str(j)
with open(path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        strategy_list.append(row['strategy'])
        taskid_list.append(int(row['taskid']))
        runid_list.append(int(row['runid']))
        num_samples_list.append(int(row['num_samples']))
        score_r2_list.append(round(float(row['score_r2']), 4))
        score_mse_list.append(round(float(row['score_mse']), 4))
        score_mae_list.append(round(float(row['score_mae']), 4))
        score_evc_list.append(round(float(row['score_evc']), 4))

path = 'logs/local_fidelity_num_samples_500'
with open(path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        strategy_list.append(row['strategy'])
        taskid_list.append(int(row['taskid']))
        runid_list.append(int(row['runid']))
        num_samples_list.append(int(row['num_samples']))
        score_r2_list.append(round(float(row['score_r2']), 4))
        score_mse_list.append(round(float(row['score_mse']), 4))
        score_mae_list.append(round(float(row['score_mae']), 4))
        score_evc_list.append(round(float(row['score_evc']), 4))

dict_res = {
    'strategy': strategy_list,
    'taskid': taskid_list,
    'runid': runid_list,
    'num_samples': num_samples_list,
    'score_r2': score_r2_list,
    'score_mse': score_mse_list,
    'score_mae': score_mae_list,
    'score_evc': score_evc_list,
}

df_res = pd.DataFrame(dict_res)

df_0 = df_res[df_res['taskid']==filtered_tasks_ids[j]]
y = 'score_mse'
# y = 'score_mae'
# y = 'score_r2'
# y = 'score_evc'
# savepath = 'logs/1.pdf'
if blackbox == 'rf':
    savepath = 'logs/rf_local_fidelity_'+y+'_'+str(j)+'.pdf'
else:
    savepath = 'logs/local_fidelity_'+y+'_'+str(j)+'.pdf'

fig, ax = plt.subplots(1, 1, figsize=(16.0, 8.0))
# sns.catplot(x='num_samples', y=y, hue='strategy',
#             kind='point', legend_out=False,
#             data=df_0,
# )
sns.relplot(x='num_samples', y=y, hue='strategy',
            kind='line',
            data=df_0,
)

plt.savefig(savepath, bbox_inches='tight')
