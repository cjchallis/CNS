import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from collections import Counter

cns_df = pandas.io.excel.read_excel("CNSdata.xlsx", "Sheet1")
columns = list(cns_df.columns.values)
for i in range(3, len(columns) - 1):
    print(columns[i])
    print(Counter(cns_df[columns[i]]))
'''    
for i in range(3, len(columns) - 1):
    for j in range(i+1, len(columns) - 1):
        print(pandas.crosstab(cns_df[columns[i]], cns_df[columns[j]]))
'''
