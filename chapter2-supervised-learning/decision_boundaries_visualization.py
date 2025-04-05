import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import mglearn

X, y = mglearn.datasets.make_forge()
print("X.shape: {}".format(X.shape))  # (26, 2) 26 samples, 2 features
print("Samples and their features:\n{}".format(X))

print("y.shape: {}".format(y.shape))  # (26,) 26 samples
print("Samples and their labels:\n{}".format(y))

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y) # I train the model for the whole dataset, not splitting it into train and test sets
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbors".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")


axes[0].legend(loc=3)

plt.savefig("decision_boundaries.png")
