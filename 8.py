import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as sm
import pandas as pd
import numpy as np

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
Y = pd.DataFrame(iris.target, columns=['Targets'])

colormap = np.array(['red', 'lime', 'black'])

plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[Y.Targets], s=40)
plt.title('Real Clustering')

model1 = KMeans(n_clusters=3)
model1.fit(X)

plt.subplot(1, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model1.labels_], s=40)
plt.title('K Means Clustering')
plt.show()

model2 = GaussianMixture(n_components=3)
model2.fit(X)

plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model2.predict(X)], s=40)
plt.title('EM Clustering')
plt.show()

print("Actual Targets:\n", iris.target)
print("KMeans Labels:\n", model1.labels_)
print("EM Labels:\n", model2.predict(X))
print("Accuracy of KMeans:", sm.accuracy_score(Y, model1.labels_))
print("Accuracy of EM:", sm.accuracy_score(Y, model2.predict(X)))
