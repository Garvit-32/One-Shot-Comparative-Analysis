import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df1 = pd.read_csv(
    'logs_files/matching_nets/omniglot_n=1_k=5_q=5_nv=1_kv=5_qv=1_dist=cosine_fce=None.csv')
df2 = pd.read_csv(
    'logs_files/matching_nets/omniglot_n=1_k=10_q=10_nv=1_kv=10_qv=1_dist=cosine_fce=None.csv')
df3 = pd.read_csv(
    'logs_files/matching_nets/omniglot_n=1_k=15_q=15_nv=1_kv=15_qv=1_dist=cosine_fce=None.csv')


plt.plot(list(range(len(df1['val_1-shot_5-way_acc'].values))),
         df1['val_1-shot_5-way_acc'].values, label='5 way')
plt.plot(list(range(len(df2['val_1-shot_10-way_acc'].values))),
         df2['val_1-shot_10-way_acc'].values, label='10 way')
plt.plot(list(range(len(df3['val_1-shot_15-way_acc'].values))),
         df3['val_1-shot_15-way_acc'].values, label='15 way')
plt.title(f'Matching Networks Omniglot')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()


plt.show()
plt.savefig('plot/omniglot_acc_match.png')
