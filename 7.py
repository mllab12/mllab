from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

iris_dataset = load_iris()

print("Target Names:", iris_dataset.target_names)
for i in range(len(iris_dataset.target_names)):
    print(f"[{i}]:[{iris_dataset.target_names[i]}]")

print("Feature Data:\n", iris_dataset["data"])

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0)

print("Target Labels:\n", iris_dataset["target"])
print("X Train:\n", X_train)
print("X Test:\n", X_test)
print("Y Train:\n", y_train)
print("Y Test:\n", y_test)

kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(X_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
print("X New:\n", x_new)
prediction = kn.predict(x_new)
print("Predicted Target Value:", prediction)
print("Predicted Class Name:", iris_dataset["target_names"][prediction][0])

for i in range(len(X_test)):
    x = X_test[i]
    x_new = np.array([x])
    prediction = kn.predict(x_new)
    print(f"Actual: {y_test[i]} {iris_dataset['target_names'][y_test[i]]}, "
          f"Predicted: {prediction[0]} {iris_dataset['target_names'][prediction][0]}")

print("Test Accuracy: {:.2f}".format(kn.score(X_test, y_test)))

