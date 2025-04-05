import mglearn
import matplotlib.pyplot as plt

mglearn.plots.plot_knn_regression(n_neighbors=1) # wave dataset
plt.savefig("knn_regression_on_wave_dataset.png")
