# Poisonous Mushroom Classification

Kaggle competition for classifying whether a mushroom is edible or poisonous based on its physical characteristics.

# Model 1
For my first model, I used light LGBM Classifier. The data was processed. This model achieved a 98.369% accuracy.

# Model 2
For the second model, I found an additional dataset which contained 60000 extra mushrooms, it wasn't very much compared to the training dataset but I still added it in. Data was preprocessed then ran through an XGB Classifier and then a LGBM Classifier. It achieved a 98.489% accuracy.

# Model 3
For the third model, I focused deeply on processing the data. 
1. Sort Data into Categorical and Numerical Columns.
2. Replace Non-Alpha Values in Categorical Columns.
3. Handle Missing Values in Categorical and Numerical Columns.
4. Replace Rare Categorical Values. (Frequency less than 100)
5. Apply Box-Cox Transformation on Skewed Numeric Columns.
6. Handle Outliers in Numeric Columns.
7. Scale Numeric Features.
8. Encode Labels.
9. Ordinal Encode Categorical Features.
10. Split Dataset into Training and Testing

Model was trained on a XGB Classifier and achieved a 98.300% accuracy.

# Model 4
For the fourth model, I branched off of the first model and included all the processing steps. I then trained the model with an XGB Classifier, an LGBM Classifier, and a CatBoost Classifier. The outcome was dissapointing as it only achieved a 98.216% accuracy. This was a let down and I started from scratch after this.

# Model 5
Model 5 was changed quite a bit overtime. Near the end I found some external data that contained a million mushrooms. Data was encoded and sorted between numerical and categorical. Created an encoder with Column Transformer applying Ordinal Encoder. StratifiedKFold with 5 splits was assigned to the variable cv. The cross_validate_score function was taken from somewhere else. It includes early stopping. If the classifier is CatBoost, Categorical features are converted to strings and missing values are filled. Classifier parameters were next. I took them from someone else who had already gone through optuna to find the best possible values for each variable. I adjusted a couple things and increased the number of estimators. Next RFECV selected the best features.

This model was run multiple times on different random states. The highest accuracy it achieved was 98.517% which was a huge step-up from the other models.