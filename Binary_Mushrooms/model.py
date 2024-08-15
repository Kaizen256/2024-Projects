import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import sys
import pandas as pd
import random
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU for computation.")
else:
    device = torch.device('cpu')
    print("No GPU available. Using CPU for computation.")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
cat_cols = ["cap-shape","cap-surface","cap-color","does-bruise-or-bleed","gill-attachment",
                "gill-spacing","gill-color","stem-root","stem-surface","stem-color","veil-type",
                "veil-color","has-ring","ring-type","spore-print-color","spore-print-color",
                "habitat","season"]
for col in cat_cols:
    train[col] = train[col].fillna('missing')
    test[col] = test[col].fillna('missing')
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
Y_train = train['class']
X_train = train.drop(labels = ["class"], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)

clf = LGBMClassifier(objective='binary', metric='binary_error',num_leaves=81,
    learning_rate=0.1, n_estimators=550, max_depth= 9, random_state=21)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_val)
matthews_corrcoef(y_val, y_pred)
y_pred = clf.predict(test)
df_sub = pd.read_csv('sample_submission.csv')
df_sub['class'] = y_pred
df_sub.to_csv('submission1.csv', index=False)

#98.369% acc