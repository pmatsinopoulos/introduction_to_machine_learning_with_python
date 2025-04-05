import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression()

# fit model on training data
lr.fit(X_train, y_train)


score_on_training_set = lr.score(X_train, y_train)
print("Training set R^2 score: {:.2f}".format(score_on_training_set)) # 0.95 which is very good

score_on_test_set = lr.score(X_test, y_test)
print("Test set R^2 score: {:.2f}".format(score_on_test_set)) # 0.61 which is not very good.
# When training set has very good score, but test set has a very low score, this discrepancy is
# a clear sign of overfitting. We need to find a model that controls complexity.
# Ridge regression is better than standard linear regression because it adds a penalty term to the loss function
