import pandas as pd
import csv

import seaborn as sns
import matplotlib.pyplot as plt

filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]
# 0, 1, 3, 4, 5, 6
j = 6
# blackbox = 'nn'
blackbox = 'rf'

title = ''
if j == 0:
    title = 'dataset name: kr-vs-kp'
elif j == 1:
    title = 'dataset name: mfeat-pixel'
elif j == 3:
    title = 'dataset name: soybean'
elif j == 4:
    title = 'dataset name: splice'
elif j == 5:
    title = 'dataset name: tic-tac-toe'
elif j == 6:
    title = 'dataset name: monks-problems-1'


# get original logs
strategy_list, taskid_list, repeatid_list, num_samples_list, featureid_list, coef_list = [], [], [], [], [], []


# related to task filtered_tasks_ids[0]
if blackbox == 'rf':
    path = 'logs/rf_coefs_stability_'+str(j)
    savepath = 'logs/rf_ae_all_coefs_stability_'+str(j)+'.pdf'
else:
    path = 'logs/coefs_stability_'+str(j)
    savepath = 'logs/ae_all_coefs_stability_'+str(j)+'.pdf'


with open(path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        strategy = row['strategy']
        if strategy == 'with embedding':
            strategy = 'entity embedding'
        elif strategy == 'with mlp embedding':
            strategy = 'mlp embedding'
        elif strategy == 'with ae mlp embedding':
            strategy = 'ae embedding'
        else:
            strategy = 'without embedding'
        strategy_list.append(strategy)

        taskid_list.append(int(row['taskid']))
        repeatid_list.append(int(row['repeatid']))
        num_samples_list.append(int(row['num_samples']))
        featureid_list.append(int(row['featureid']))
        coef_list.append(float(row['coef']))
        # score_evc_list.append(round(float(row['score_evc']), 4))

if blackbox == 'rf':
    path = 'logs/rf_mlp_coefs_stability_'+str(j)
else:
    path = 'logs/mlp_coefs_stability_'+str(j)
with open(path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        strategy = row['strategy']
        if strategy == 'with embedding':
            strategy = 'entity embedding'
        elif strategy == 'with mlp embedding':
            strategy = 'mlp embedding'
        elif strategy == 'with ae mlp embedding':
            strategy = 'ae embedding'
        else:
            strategy = 'without embedding'
        strategy_list.append(strategy)

        taskid_list.append(int(row['taskid']))
        repeatid_list.append(int(row['repeatid']))
        num_samples_list.append(int(row['num_samples']))
        featureid_list.append(int(row['featureid']))
        coef_list.append(float(row['coef']))
        # score_evc_list.append(round(float(row['score_evc']), 4))

if blackbox == 'rf':
    path = 'logs/rf_ae_mlp_coefs_stability_'+str(j)
else:
    path = 'logs/ae_mlp_coefs_stability_'+str(j)
with open(path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        strategy = row['strategy']
        if strategy == 'with embedding':
            strategy = 'entity embedding'
        elif strategy == 'with mlp embedding':
            strategy = 'mlp embedding'
        elif strategy == 'with ae mlp embedding':
            strategy = 'ae embedding'
        else:
            strategy = 'without embedding'
        strategy_list.append(strategy)

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
    df_entity_emb = df_res[df_res['strategy'] == 'entity embedding']
    df_entity_emb = df_entity_emb[df_entity_emb['num_samples'] == num_samples]
    entity_emb_coef_std = df_entity_emb.groupby(['featureid'])['coef'].std()
    strategy_list.append('entity embedding')
    num_samples_list.append(num_samples)
    coef_avgstd_list.append(entity_emb_coef_std.mean())

    df_without_emb = df_res[df_res['strategy'] == 'without embedding']
    df_without_emb = df_without_emb[df_without_emb['num_samples'] == num_samples]
    without_emb_coef_std = df_without_emb.groupby(['featureid'])['coef'].std()
    strategy_list.append('without embedding')
    num_samples_list.append(num_samples)
    coef_avgstd_list.append(without_emb_coef_std.mean())

    df_mlp_emb = df_res[df_res['strategy'] == 'mlp embedding']
    df_mlp_emb = df_mlp_emb[df_mlp_emb['num_samples'] == num_samples]
    mlp_emb_coef_std = df_mlp_emb.groupby(['featureid'])['coef'].std()
    strategy_list.append('mlp embedding')
    num_samples_list.append(num_samples)
    coef_avgstd_list.append(mlp_emb_coef_std.mean())

    df_ea_emb = df_res[df_res['strategy'] == 'ae embedding']
    df_ea_emb = df_ea_emb[df_ea_emb['num_samples'] == num_samples]
    ae_emb_coef_std = df_ea_emb.groupby(['featureid'])['coef'].std()
    strategy_list.append('ae embedding')
    num_samples_list.append(num_samples)
    coef_avgstd_list.append(ae_emb_coef_std.mean())

dict_stats = {
    'strategy': strategy_list,
    'num_samples': num_samples_list,
    'coef_avgstd': coef_avgstd_list,
}
df_stats = pd.DataFrame(dict_stats)


fig, ax = plt.subplots(1, 1, figsize=(16.0, 8.0))
sns.catplot(x='num_samples', y='coef_avgstd', hue='strategy',
            # markers=["^", "o"], linestyles=["-", "--"],
            kind='point', data=df_stats, legend_out=False,
)
plt.title(title)
plt.savefig(savepath, bbox_inches='tight')

