from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
for i in range(len(X_test)):
    pred = model.predict([X_test[i]])
    print("Actual:", y_test[i], data.target_names[y_test[i]], 
          "Predicted:", pred[0], data.target_names[pred[0]])
print("Accuracy:", model.score(X_test, y_test))
