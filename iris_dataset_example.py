from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt # Importing matplotlib for plotting
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# We will check that the model does good job on the test set
# We can use the pair plot to see the relationship between the features
# We will use the pandas library to create a dataframe from the data in X_train
# The dataframe will be used to create a scatter matrix
# The scatter matrix will color by y_train
# The scatter matrix will be saved as scatter_matrix.png

# create dataframe from data in X_train (this is the iris training set)
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.savefig("scatter_matrix.png")
# plt.show()

# We will use the KNeighborsClassifier to build the model. This model
# will be used to classify the iris species.
# The model will be trained using the training data
# The model will be evaluated using the test data

knn = KNeighborsClassifier(n_neighbors=1)
# -- TRAIN --
# We use the train data to train the model
knn.fit(X_train, y_train)

print("{}".format(knn))

# -- PREDICTION --
# Let's suppose that we have some new data for which we need to predict the species
# We will use the predict method of the knn object to make predictions
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# -- EVALUATION --
# But how can we trust the model?
# We will use the test data to evaluate the model
y_pred = knn.predict(X_test)
print("Test set predictions:\n{}".format(y_pred))
print("Test set prediction name: {}".format(iris_dataset['target_names'][y_pred]))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
