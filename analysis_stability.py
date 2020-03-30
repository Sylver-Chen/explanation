import pandas as pd
import csv

import seaborn as sns
import matplotlib.pyplot as plt

filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]
j = 12

# get original logs
strategy_list, taskid_list, repeatid_list, num_samples_list, featureid_list, coef_list = [], [], [], [], [], []

# related to task filtered_tasks_ids[0]
logpath = 'logs/coefs_stability_allfeats_'+str(j)
savepath = 'logs/stability_allfeats_'+str(j)+'.pdf'

with open(logpath, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        strategy_list.append(row['strategy'])
        taskid_list.append(int(row['taskid']))
        repeatid_list.append(int(row['repeatid']))
        num_samples_list.append(int(row['num_samples']))
        featureid_list.append(int(row['featureid']))
        coef_list.append(float(row['coef']))
        # score_evc_list.append(round(float(row['score_evc']), 4))

dict_res = {
    'strategy': strategy_list,
    'taskid': taskid_list,
    'repeatid': repeatid_list,
    'num_samples': num_samples_list,
    'featureid': featureid_list,
    'coef': coef_list,
}
df_res = pd.DataFrame(dict_res)

# preprocess logs
strategy_list, num_samples_list, coef_avgstd_list = [], [], []
num_samples_iter = list(range(500, 5500, 500))
for num_samples in num_samples_iter:
    df_with_emb = df_res[df_res['strategy'] == 'with embedding']
    df_with_emb = df_with_emb[df_with_emb['num_samples'] == num_samples]
    with_emb_coef_std = df_with_emb.groupby(['featureid'])['coef'].std()
    strategy_list.append('with embedding')
    num_samples_list.append(num_samples)
    coef_avgstd_list.append(with_emb_coef_std.mean())

    df_without_emb = df_res[df_res['strategy'] == 'without embedding']
    df_without_emb = df_without_emb[df_without_emb['num_samples'] == num_samples]
    without_emb_coef_std = df_without_emb.groupby(['featureid'])['coef'].std()
    strategy_list.append('without embedding')
    num_samples_list.append(num_samples)
    coef_avgstd_list.append(without_emb_coef_std.mean())

dict_stats = {
    'strategy': strategy_list,
    'num_samples': num_samples_list,
    'coef_avgstd': coef_avgstd_list,
}
df_stats = pd.DataFrame(dict_stats)


fig, ax = plt.subplots(1, 1, figsize=(16.0, 8.0))
sns.catplot(x='num_samples', y='coef_avgstd', hue='strategy',
            markers=["^", "o"], linestyles=["-", "--"],
            kind='point', data=df_stats, legend_out=False,
)
plt.savefig(savepath, bbox_inches='tight')

