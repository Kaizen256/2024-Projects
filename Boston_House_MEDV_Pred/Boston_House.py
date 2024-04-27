import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


numerical_feat = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_feat])
target = df.MEDV


inputs_train, inputs_test, targets_train, targets_test = train_test_split(scaled_data, target, test_size=0.1)


model = LinearRegression().fit(inputs_train, targets_train)
predictions_test = model.predict(inputs_test)


loss = rmse(targets_test, predictions_test)
print('Loss:', loss)


plt.figure(figsize=(8, 6))
plt.scatter(targets_test, predictions_test, color='blue')
plt.plot(targets_test, targets_test, color='red', linestyle='--')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()
