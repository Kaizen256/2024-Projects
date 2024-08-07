from urllib.request import urlretrieve
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
urlretrieve(medical_charges_url, 'medical.csv')
medical_df = pd.read_csv('medical.csv')


def try_parameters(w, b):
    ages = medical_df.age
    target = medical_df.charges
    predictions = estimate_charges(ages, w, b)


    plt.plot(ages, predictions, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8, alpha=0.9)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.show()


    loss = rmse(target, predictions)
    print(f"RMSE LOSS: {loss}")


def estimate_charges(age, w, b):
    return w * age + b


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


#non numerical
smoker_codes = {'no': 0, 'yes': 1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)
#non numerical
sex_codes = {'female': 0, 'male': 1}
medical_df['sex_code'] = medical_df.sex.map(sex_codes)


#multiple values
enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[['region']])
one_hot = enc.fit_transform(medical_df[['region']]).toarray()
medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot


#scaling numerical columns:
numeric_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
scaled_inputs = scaler.fit_transform(medical_df[numeric_cols])


cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = medical_df[cat_cols].values


inputs = np.concatenate((scaled_inputs, categorical_data), axis=1)
targets = medical_df.charges


#creating test set
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.1)


#Create and train the model
model = LinearRegression().fit(inputs_train, targets_train)


#generate predictions
predictions_test = model.predict(inputs_test)


#compute loss to evaluate
loss = rmse(targets_test, predictions_test)
print('Loss: ', loss)

plt.figure(figsize=(8, 6))
plt.scatter(targets_test, predictions_test, color='blue')
plt.plot(targets_test, targets_test, color='red', linestyle='--')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()