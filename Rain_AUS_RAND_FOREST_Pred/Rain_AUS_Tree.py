import opendatasets as od
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

od.download('https://www.kaggle.com/jsphyg/weather-dataset-rattle-package')
os.listdir('weather-dataset-rattle-package')
raw_df = pd.read_csv('weather-dataset-rattle-package/weatherAUS.csv')

raw_df.dropna(subset=['RainTomorrow'], inplace=True)

year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

imputer = SimpleImputer(strategy = 'mean').fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

scaler = StandardScaler().fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(raw_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

#model = DecisionTreeClassifier(random_state=42) IF THIS OVERFITS USE:
#IF YOU WANT TO GO IN DEPTH WITH MAX_DEPTH AN MAX_LEAF_NODES TO PREVENT OVERFITTING SHIT:
#-------------------------------------------------------------------------
#IF YOU WANT TO FIND MAX_DEPTH ERROR IN ORDER TO MAKE SEARCHING FOR LEAF NODES ERROR EASIER:
def max_depth_error(md):
    model = DecisionTreeClassifier(max_depth=md, random_state=42)
    model.fit(X_train, train_targets)
    train_acc = 1 - model.score(X_train, train_targets)
    val_acc = 1 - model.score(X_val, val_targets)
    return {'Max Depth': md, 'Training Error': train_acc, 'Validation Error': val_acc}
#errors_df = pd.DataFrame([max_depth_error(md) for md in range(1, 21)])

#THEN FIND LEAF_NODES:
def leaf_nodes_error(mln):
    model = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=mln, random_state=42)
    model.fit(X_train, train_targets)
    train_acc = 1 - model.score(X_train, train_targets)
    val_acc = 1 - model.score(X_val, val_targets)
    return {'Max Depth': 7, 'Max Leaf Nodes': mln, 'Training Error': train_acc, 'Validation Error': val_acc}

# Generate a grid of max_depth and max_leaf_nodes values
#max_depth_values = range(6, 8)
#max_leaf_nodes_values = [10, 50, 100, 200, None]

#errors_df = pd.DataFrame([leaf_nodes_error(mln) for mln in max_leaf_nodes_values])
#print(errors_df)
#-------------------------------------------------------------------------

#SUBPAR 84.53949277% MODEL
#model = DecisionTreeClassifier(max_depth=7, max_leaf_nodes = 200, random_state=42)
#model.fit(X_train, train_targets)
#PLOTS DECISION TREE DIAGRAM IF NECCESSARY
#plt.figure(figsize=(20,10))
#plot_tree(model, feature_names=X_train.columns, max_depth=2, filled=True)
#plt.show()

#-------------------------------------------------------------------------
#IMPORTANCE OF DIFFERENT VARIABLES
#importance_df = pd.DataFrame({
#   'feature': X_train.columns,
#    'importance': model.feature_importances_
#}).sort_values('importance', ascending=False)

#print(importance_df.head(10))
#PLOT THIS SHIT IF YOU WANT
#plt.title('Feature Importance')
#sns.barplot(data=importance_df.head(10), x='importance', y='feature')
#plt.show()
#-------------------------------------------------------------------------

#TRAIN RANDOM FOREST CLASSIFIER TO SEE IF ITS FASTER 0.856% MODEL SPEED
#model = RandomForestClassifier(n_jobs=-1, random_state=42)
#model.fit(X_train, train_targets)

def test_params(**params):
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params).fit(X_train, train_targets)
    return model.score(X_train, train_targets), model.score(X_val, val_targets)
print(test_params(max_depth=26))
#85.57% acc validation 98.15% acc train



