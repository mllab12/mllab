import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score

df = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/New folder/newclass/trial/seattle-weather.csv")
df = df.drop(['date'], axis=1)

le = LabelEncoder()
df['w'] = le.fit_transform(df['weather'])
df = df.drop(['weather'], axis=1)

X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = df['w']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)

model = LinearRegression()
model.fit(X_train, y_train)

sample_prediction = model.predict([[8.6, 4.4, 1.7, 1.3]])
print("\nSample Prediction (encoded class):", round(sample_prediction[0]))

y_pred = model.predict(X_test).round(0)
y_test = y_test.round(0)

f1 = f1_score(y_test, y_pred, average='micro')
print("\nF1 Score:", f1)

indices = np.arange(50)
plt.figure(figsize=(10, 5))
plt.scatter(indices, y_pred[:50], color='red', label='Predicted')
plt.scatter(indices, y_test[:50], color='blue', label='Actual')
plt.xlabel("Sample Index")
plt.ylabel("Encoded Weather Class")
plt.title("Actual vs Predicted Weather Conditions (First 50 Samples)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
