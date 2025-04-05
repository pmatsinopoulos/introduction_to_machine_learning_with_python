from sklearn.linear_model import LinearRegression
import mglearn
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression()

# train model
lr.fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_)) # slope of the line
print("lr.intercept_: {}".format(lr.intercept_)) # y intercept of the line, the bias term

# How can I look at the training set performance?
score_on_training_set = lr.score(X_train, y_train) # R^2 score of the model on the training set)
print("Training set R^2 score: {:.2f}".format(score_on_training_set))

score_on_test_set = lr.score(X_test, y_test) # 0.66 which is not very good
print("Test set R^2 score: {:.2f}".format(score_on_test_set)) # 0.69 which is not very good
# Scores are very closed together. This means that we are likely underfitting the model not overfitting it.
