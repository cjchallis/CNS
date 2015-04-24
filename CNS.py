import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

cns_df = pandas.io.excel.read_excel("CNSdata.xlsx", "Sheet1")
print(cns_df.pivot_table(rows = "Head", cols = "Arms"))
