import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
from collections import Counter

cns_df = pd.read_csv('CNS data.csv')
columns = list(cns_df.columns.values)

# print and save original frequencies
freq = open('freq.txt', 'w')
row = "{:<%d}{:>%d}" % (20, 5)
for i in range(3, len(columns)):
    print(columns[i])
    freq.write(columns[i] + '\n')
    d = dict(Counter(cns_df[columns[i]]))
    print(d)
    for key in d:
        freq.write(row.format(str(key) + ':', d[key]))
        freq.write('\n')
    freq.write('\n')

# ----------------------------------------------------------------------------- 
# Clean data
# ----------------------------------------------------------------------------- 

# many values have extra white space
for col in columns:
    cns_df[col] = cns_df[col].str.strip()
# replace remaining blanks with nan
cns_df = cns_df.replace('', np.nan)
cns_df = cns_df.replace('Yes', 'yes')
cns_df = cns_df.replace('No', 'no')
cns_df = cns_df.replace('yes?', 'yes')
cns_df = cns_df.replace('Crossed', 'crossed')
cns_df = cns_df.replace('Extended', 'extended')

# manually replace inconsistent values for each column
cns_df['Class'] = cns_df['Class'].str.replace('Ppr', 'P')
cns_df['Class'] = cns_df['Class'].str.replace('p', 'P')
cns_df['Class'] = cns_df['Class'].str.replace('T/S', 'T')

cns_df['Position'] = cns_df['Position'].str.replace('C Dorsa$', 'C Dorsal')

cns_df['Head'] = cns_df['Head'].str.replace('to east', 'east')

cns_df['Arms'] = cns_df['Arms'].str.replace('Flexed', 'flexed') 
cns_df['Arms'] = cns_df['Arms'].str.replace('flezed', 'flexed') 

cns_df['Legs'] = cns_df['Legs'].str.replace('extendede', 'extended')

cns_df['N transforms'] = cns_df['N transforms'].str.replace('bug$', 'bugs')

cns_df['Sex'] = cns_df['Sex'].str.replace('s', 'S')


print(chi2_contingency(pd.crosstab(cns_df['Class'], cns_df['Age']))[0])

row = columns[3:]
col = []
for val in row:
    col.append(val + '_c')

corr = pd.DataFrame(0.00, index=row, columns=col)
print(corr['Class_c']['Class'])

for r in row:
    for c in col:
        if r != c[:-2]:
            tab = pd.crosstab(cns_df[r], cns_df[c[:-2]])
            chi2 = chi2_contingency(tab)[0]
            n = tab.sum().sum()
            min_d = min(tab.shape)-1
            corr[c][r] = (chi2 / n /min_d) ** 0.5

sns.set(style="white")

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
2
f, ax = plt.subplots(figsize=(11,9))
xticks = corr.columns
yticks = corr.index
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
hm = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, square=True, xticklabels=corr.index,
            yticklabels=True, linewidths=.5, cbar=False, ax=ax)
plt.setp(hm.get_yticklabels(), rotation=0)

plt.show()
