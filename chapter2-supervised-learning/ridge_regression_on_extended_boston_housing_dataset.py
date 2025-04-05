import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

rr = Ridge(alpha=1.0)  # alpha is the regularization strength

rr.fit(X_train, y_train)

score_on_training_set = rr.score(X_train, y_train)
print("Training set R^2 score: {:.2f}".format(score_on_training_set)) # 0.89

score_on_test_set = rr.score(X_test, y_test)
print("Test set R^2 score: {:.2f}".format(score_on_test_set)) # 0.75

# This is much less discrepancy between training and test set scores.
