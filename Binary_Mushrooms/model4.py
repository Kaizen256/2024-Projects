import warnings
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import gc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import mode

warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
train = train.drop(['id'], axis=1)
train2 = pd.read_csv('secondary_data.csv', sep=";")
df_train = pd.concat([train, train2], ignore_index=True)
df_train = df_train.drop_duplicates(subset=df_train.columns.to_list().remove('class'), keep='first')
test = pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')

categorical_columns = df_train.select_dtypes(include=['object']).columns
categorical_columns2 = test.select_dtypes(include=['object']).columns
numeric_columns = df_train.select_dtypes(include=[np.number]).columns
numeric_columns2 = df_train.select_dtypes(include=[np.number]).columns

def rep_non_alpha(value):
    if isinstance(value, str) and (len(value) == 1 and value.isalpha()):
        return value
    return np.nan

for col in categorical_columns:
    df_train[col] = df_train[col].apply(rep_non_alpha)

for col in categorical_columns2:
    test[col] = test[col].apply(rep_non_alpha)

for col in categorical_columns:
    if df_train[col].isnull().any():
        mode_value = df_train[col].mode()[0] if not df_train[col].mode().empty else 'Unknown'
        df_train[col].fillna(mode_value, inplace=True)

for col in categorical_columns2:
    if test[col].isnull().any():
        mode_value = test[col].mode()[0] if not test[col].mode().empty else 'Unknown'
        test[col].fillna(mode_value, inplace=True)

for col in numeric_columns:
    if df_train[col].isnull().any():
        median_value = df_train[col].median()
        df_train[col].fillna(median_value, inplace=True)

for col in numeric_columns2:
    if test[col].isnull().any():
        median_value = test[col].median()
        test[col].fillna(median_value, inplace=True)

mappings = {}
threshold = 100

for col in categorical_columns:
    value_counts = df_train[col].value_counts()
    values_to_replace = value_counts[value_counts < threshold].index
    mappings[col] = {value: 'Unknown' for value in values_to_replace}
    df_train[col] = df_train[col].replace(values_to_replace, 'Unknown')

for col in categorical_columns2:
    value_counts = test[col].value_counts()
    values_to_replace = value_counts[value_counts < threshold].index
    mappings[col] = {value: 'Unknown' for value in values_to_replace}
    test[col] = test[col].replace(values_to_replace, 'Unknown')

threshold = 1.0
lambda_train = {}
lambda_test = {}
for col in numeric_columns:
    skewness = df_train[col].skew()
    if skewness > threshold:
        df_train[col] = df_train[col] + 1
        df_train[col], fitted_lambda = boxcox(df_train[col])
        lambda_train[col] = fitted_lambda
    skewness2 = test[col].skew()
    if skewness > threshold:
        test[col] = test[col] + 1
        test[col], fitted_lambda = boxcox(test[col])
        lambda_test[col] = fitted_lambda

for column in df_train.select_dtypes(include=['number']).columns:
    Q1 = df_train[column].quantile(0.25)
    Q3 = df_train[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_train[column] = df_train[column].clip(lower=lower_bound, upper=upper_bound)

    Q2 = test[column].quantile(0.25)
    Q4 = test[column].quantile(0.75)
    IQR2 = Q4 - Q2
    lower_bound = Q2 - 1.5 * IQR2
    upper_bound = Q4 + 1.5 * IQR2
    test[column] = test[column].clip(lower=lower_bound, upper=upper_bound)

numeric_features = ['cap-diameter', 'stem-height', 'stem-width']

X_train_numeric = df_train[numeric_features]
X_test_numeric = test[numeric_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

df_train[numeric_features] = X_train_scaled
test[numeric_features] = X_test_scaled

label = df_train['class']

label_encoder = LabelEncoder()
df_train['class'] = label = label_encoder.fit_transform(df_train['class'])

cat_cols_train = df_train.select_dtypes(include=['object']).columns
cat_cols_train = cat_cols_train[cat_cols_train != 'class']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

df_train[cat_cols_train] = ordinal_encoder.fit_transform(df_train[cat_cols_train].astype(str))
test[cat_cols_train] = ordinal_encoder.transform(test[cat_cols_train].astype(str))

X = df_train.drop(['class'], axis=1)
X_big, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42,
                                                  shuffle=True, stratify=df_train['class'])

xgb_optuna_params = {
    'n_estimators': 2500,
    'alpha': 0.0002,
    'subsample': 0.60,
    'colsample_bytree': 0.4,
    'max_depth': 13,
    'min_child_weight': 10,
    'learning_rate': 0.002,
    'gamma': 5.6e-08,
    'device': "cuda"
}

lgb_params = {
    'n_estimators': 2500,
    'max_bin': 256,
    'colsample_bytree': 0.6,
    'reg_lambda': 80,
    'n_jobs': -1
}

catb_params = {
    'n_estimators': 2500,
    'learning_rate': 0.01,
    'depth': 10,
    'l2_leaf_reg': 3,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'verbose': 0,
    'random_seed': 42
}

gc.collect()
xgb_models = [(f"xgb_{i}", XGBClassifier(**xgb_optuna_params, random_state=i)) for i in range(9)]
lgb_models = [(f"lgb_{i}", LGBMClassifier(**lgb_params, random_state=i)) for i in range(12)]
catb_models = [(f"cat_{i}", CatBoostClassifier(**catb_params, random_state=i)) for i in range(10)]

for name, model in xgb_models:
    print(f"Training {name}...")
    model.fit(X_big, y_train, eval_set=[(X_test, y_test)], verbose=200)

for name, model in lgb_models:
    print(f"Training {name}...")
    model.fit(X_big, y_train, eval_set=[(X_test, y_test)], verbose=200)

for name, model in catb_models:
    print(f"Training {name}...")
    model.fit(X_big, y_train, eval_set=[(X_test, y_test)], verbose=200)

all_preds = []

for name, model in xgb_models:
    y_pred_xgb = model.predict(X_test)
    test_pred_xgb = model.predict(test.drop('id', axis=1))
    all_preds.append(test_pred_xgb)

for name, model in lgb_models:
    y_pred_lgb = model.predict(X_test)
    test_pred_lgb = model.predict(test.drop('id', axis=1))
    all_preds.append(test_pred_lgb)

for name, model in catb_models:
    y_pred_catb = model.predict(X_test)
    test_pred_catb = model.predict(test.drop('id', axis=1))
    all_preds.append(test_pred_catb)

final_test_pred = mode(all_preds, axis=0)[0].flatten()

test_pred_labels = label_encoder.inverse_transform(final_test_pred)

sub['class'] = test_pred_labels
sub.to_csv('submission4.csv', index=False)