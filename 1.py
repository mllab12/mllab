import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


dataset = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/New folder/newclass/trial/Advertising.csv")

# Feature and target selection
X = dataset[['TV']]           
y = dataset['Sales']         


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


model = LinearRegression()
model.fit(X_train, y_train)


print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Make predictions on the test set
y_pred = model.predict(X_test)
print("\nPredictions on test set:\n", y_pred)

# Compare actual vs predicted values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted:\n", comparison.head())

# Plot actual vs predicted
plt.scatter(X_test, y_test, label="Actual", color="blue")
plt.plot(X_test, y_pred, color='red', label="Regression Line")
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.legend()
plt.title("TV Ads vs Sales - Linear Regression")
plt.show()

# Evaluate model performance
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(X, y) * 100

print("\nModel Evaluation:")
print("R-squared: {:.2f}%".format(r2))
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
