import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import matthews_corrcoef
import sklearn
sklearn.set_config(transform_output="pandas")

train = pd.read_csv('train.csv')
train2 = pd.read_csv('secondary_data.csv', sep=";")
train = pd.concat([train, train2], ignore_index=True)
cols = train.columns.to_list()
cols.remove("class")
train = train.drop_duplicates(subset=cols, keep='first')
X_test = pd.read_csv('test.csv')
sub_fl = pd.read_csv('sample_submission.csv', index_col=["id"])

def cleaning(df):
    threshold = 100
    cat_feats = ["cap-shape","cap-surface","cap-color","does-bruise-or-bleed","gill-attachment",
                 "gill-spacing","gill-color","stem-root","stem-surface","stem-color","veil-type",
                 "veil-color","has-ring","ring-type","spore-print-color","habitat","season"]
    
    for feat in cat_feats:
        df[feat] = df[feat].fillna('missing')
        df.loc[df[feat].value_counts(dropna=False)[df[feat]].values < threshold, feat] = "noise"
        df[feat] = df[feat].astype('category')
    
    return df

train = cleaning(train)
X_test = cleaning(X_test)

X = train.drop(["class"], axis="columns")
y = train["class"].map({'e': 0, 'p': 1})

X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

lgb_params = {
    'n_estimators': 2500,
    'max_bin': 256,
    'colsample_bytree': 0.6,
    'reg_lambda': 80,
    'verbose': -1,
    'n_jobs': -1
}

xgb_params = {
    'n_estimators': 2500,
    'max_bin': 1024,
    'colsample_bytree': 0.6,
    'reg_lambda': 80,
    'verbosity': 0,
    'use_label_encoder': False,
    'n_jobs': -1
}

xgb_models = [(f"xgb_{i}", XGBClassifier(**xgb_params, random_state=i)) for i in range(9)]
lgb_models = [(f"lgb_{i}", LGBMClassifier(**lgb_params, random_state=i)) for i in range(16)]

all_models = lgb_models + xgb_models

test_pred_probas = []

for name, model in all_models:
    model.fit(X, y)
    train_preds = model.predict(X)
    mcc = matthews_corrcoef(y, train_preds)
    print(f'Training Data -> Model: {name}, MCC: {round(mcc, 5)}')
    print()
    test_pred_probas.append(model.predict_proba(X_test)[:, 1])

mean_test_pred_probas = np.mean(test_pred_probas, axis=0)

threshold = 0.5
test_predictions = mean_test_pred_probas > threshold

submission = pd.read_csv("sample_submission.csv")
submission["class"] = test_predictions.astype(int)
submission['class'] = submission['class'].map({0: 'e', 1: 'p'})
submission.to_csv('submission2.csv', index=False)

#98.489