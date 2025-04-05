from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=40)

# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)

# fit the model to the training data and training targets/labels
reg.fit(X_train, y_train)

# make predictions on the test set
predictions = reg.predict(X_test)
print("Test set predictions: \n{}".format(predictions))

# let's evaluate the model on the test set
score = reg.score(X_test, y_test) # this does predictions on all X_test samples and compares them to the y_test labels
print("Test set R^2 score: {:.2f}".format(score)) # coefficient of determination is a measure of how well a regression model predicts and
# yields a score between 0 and 1. A value of 1 corresponds to a perfect prediction, while a value of 0 indicates that the model
# does not predict better than the mean of the target values. Value 0 corresponds to a constant model that
# just predicts the mean of the training set responses, y_train.
