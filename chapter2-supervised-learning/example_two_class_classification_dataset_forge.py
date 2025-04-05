import mglearn
import matplotlib.pyplot as plt

# generate dataset. X is the data, y is the labels/classes
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
# plt.show()
plt.savefig("forge_dataset.png")
print("X.shape: {}".format(X.shape)) # (26, 2), i.e. 26 samples and 2 features
print("y.shape: {}".format(y.shape)) # (26,), i.e. 26 samples

mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.savefig("knn_classification.png")
