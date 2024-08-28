import numpy as np
import pandas as pd 
import seaborn as sns
import missingno as msno
from sklearn import set_config
from sklearn.base import clone
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from scipy.stats import randint, mode
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix, matthews_corrcoef, make_scorer
import time
random_state = 67

train = pd.read_csv('train.csv', index_col=[0])
train2 = pd.read_csv('one_million_mushrooms.csv', sep=";")
train_df = pd.concat([train, train2], ignore_index=True)
test_df = pd.read_csv('test.csv', index_col=[0])

label_encoder = LabelEncoder()
train_df['class'] = label_encoder.fit_transform(train_df['class'])
feature_list = [feature for feature in train_df.columns if not feature  == "class"]
target = "class"
numerical_features = ['stem-height', 'cap-diameter', 'stem-width']
categorical_features = list(set(feature_list) - set(numerical_features))
assert feature_list.sort() == (numerical_features + categorical_features).sort()
eda_df = train_df.sample(frac= 0.1, random_state=random_state)

train_df[categorical_features] = train_df[categorical_features].astype('category')
test_df[categorical_features] = test_df[categorical_features].astype('category')

def preprocess_catboost(train_df, test_data, cat_features):
    for col in cat_features:
        train_df[col] = train_df[col].astype(str).fillna('NaN')
        test_data[col] = test_data[col].astype(str).fillna('NaN')
    return train_df, test_data

encoder  = ColumnTransformer(remainder='passthrough',
    transformers=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
    ])

y = train_df['class']
train_df = train_df.drop(['class'], axis=1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

def cross_validate_score(model, train_df, y, cv, test_data):
    val_scores = []
    test_preds = np.zeros((test_data.shape[0],))
    oof_preds = np.zeros((train_df.shape[0],))

    if isinstance(model, CatBoostClassifier):
        cat_features = model.get_params().get('cat_features', [])
        train_df, test_data = preprocess_catboost(train_df, test_data, cat_features)

    for fold, (train_idx, val_idx) in enumerate(cv.split(train_df, y)):
        X_train = train_df.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        
        X_val = train_df.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        model = clone(model)
        eval_set = [(X_val, y_val)]

        if isinstance(model, LGBMClassifier):
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=[early_stopping(50)],
            )
        elif isinstance(model, CatBoostClassifier):
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            model.fit(
                X_train, y_train,

            )
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)
        val_score = matthews_corrcoef(y_val, val_preds)
        print(f'Fold {fold}: MCC = {val_score:.5f}')
        val_scores.append(val_score)
        oof_preds[val_idx] = val_probs
        test_preds += model.predict_proba(test_data)[:, 1] / cv.get_n_splits()  # Aggregate test probabilities

    mean_val_score = np.mean(val_scores)
    std_val_score = np.std(val_scores)
    print(f'Mean Validation MCC: {mean_val_score:.7f}')
    print(f'Std Validation MCC: {std_val_score:.7f}')
    
    return val_scores, test_preds, oof_preds

cv_summary, test_preds, oof_preds = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

xgb_optuna_params = {
 'tree_method': 'gpu_hist',
 'n_estimators': 5088,
 'alpha': 4.956752183261538e-07,
 'subsample': 0.7349948172684168,
 'colsample_bytree': 0.30171411525842506,
 'max_depth': 15, 
 'min_child_weight': 6,
 'learning_rate': 0.013301072238797047,
 'gamma': 5.634602153104516e-08
}

xgb_tuned = XGBClassifier(**xgb_optuna_params, random_state=random_state)
xgb_pipeline = make_pipeline(encoder, xgb_tuned)\

cv_summary['xgb'], test_preds['xgb'], oof_preds['xgb'] = cross_validate_score(xgb_pipeline, train_df , y,  cv, test_df)

lgbm_optuna_params = {
    'n_estimators': 20000,
    'learning_rate': 0.02,
    "categorical_feature" : categorical_features,
    'max_depth': 10,
    'min_data_in_leaf': 85,
    'subsample': 0.6720606456166781,
    'max_bin': 240,
    'feature_fraction': 0.6946327643448142,
}

lgbm_tuned = LGBMClassifier(**lgbm_optuna_params, random_state=random_state, verbose=-1)
cv_summary['lgbm'], test_preds['lgbm'], oof_preds['lgbm'] = cross_validate_score(lgbm_tuned, train_df , y,  cv, test_df)

catb_params = {    
    "n_estimators" : 20000,
    "learning_rate" : 0.075,
    'cat_features' : categorical_features,
    'task_type': 'GPU',
    'random_strength': 0.3718364180573207,
    'max_bin': 128,
    'depth': 9,
    'l2_leaf_reg': 6,
    'grow_policy': 'SymmetricTree',
    'boosting_type': 'Plain',
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.41936688658110405
}

catb_tunned = CatBoostClassifier(**catb_params, random_state=random_state)
cv_summary['catb'], test_preds['catb'], oof_preds['catb'] = cross_validate_score(catb_tunned, train_df , y,  cv, test_df)

meta_model_params = {
    'C': 0.000237302749626327,
    'max_iter': 2500,
    'tol': 9.996751434702547e-05,
    'solver': 'saga',
    'penalty': 'l1'
}

meta_model = LogisticRegression(**meta_model_params, random_state=random_state)

min_features_to_select = 1

pipeline = Pipeline([
    
    ('Scaler', StandardScaler()),
    ('rfecv', RFECV(estimator=meta_model,
                    step=1,
                    cv=cv,
                    scoring=make_scorer(matthews_corrcoef),
                    min_features_to_select=min_features_to_select,
                    n_jobs=-1,))
])

pipeline.fit(oof_preds, y)

print("Best CV score: ")
selected_models = np.array( oof_preds.columns)[pipeline.named_steps['rfecv'].support_]
print( pipeline.named_steps['rfecv'].cv_results_["mean_test_score"][len(selected_models) - 1])


print('Number of available models:', len(oof_preds.columns))
print('Number of selected models for ensemble:', len(selected_models))
print("Selected models:", selected_models)

meta_model = meta_model.fit(oof_preds[selected_models], y)
preds_test =  meta_model.predict(test_preds[selected_models])
preds_test = label_encoder.inverse_transform(preds_test)
output = pd.DataFrame({'id': test_df.index,
                       'class': preds_test})
output.to_csv('submission12.csv', index=False)