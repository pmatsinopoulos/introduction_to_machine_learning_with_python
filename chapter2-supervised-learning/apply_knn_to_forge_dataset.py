from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mglearn

X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)

# Now we fit the classifier using the training set.
clf.fit(X_train, y_train)

# Make predictions on the test data in order to evaluate the classifier.
y_pred = clf.predict(X_test)
print("Test set predictions: \n{}".format(y_pred))

# Evaluate how well our classifier generalizes to unseen data.
score = clf.score(X_test, y_test)
print("Test set score: {:.2f}".format(score))
