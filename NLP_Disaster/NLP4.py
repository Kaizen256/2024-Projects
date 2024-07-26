import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string

train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
sub_sample = pd.read_csv("Data/sample_submission.csv")
#Unnecessary but I want it
train = train.drop_duplicates().reset_index(drop=True)

kw_d = train[train.target==1].keyword.value_counts().head(10)
kw_nd = train[train.target==0].keyword.value_counts().head(10)
top_d = train.groupby('keyword').mean()['target'].sort_values(ascending=False).head(10)
top_nd = train.groupby('keyword').mean()['target'].sort_values().head(10)
raw_loc = train.location.value_counts()
top_loc = list(raw_loc[raw_loc>=10].index)
top_only = train[train.location.isin(top_loc)]