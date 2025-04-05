import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import numpy as np

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso = Lasso(alpha=0.01, max_iter=100_000)
# lower alpha means a more complex model which works better on the training and test data.

lasso.fit(X_train, y_train)

score_on_training_set = lasso.score(X_train, y_train)
print("Training set R^2 score: {:.2f}".format(score_on_training_set))  # 0.95

score_on_test_set = lasso.score(X_test, y_test)
print("Test set R^2 score: {:.2f}".format(score_on_test_set)) # 0.64

print("Number of total features: {}".format(len(lasso.coef_))) # 104
print("Number of features in the training set: {}".format(X_train.shape[1]))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0))) # 96

# this is the best alpha
