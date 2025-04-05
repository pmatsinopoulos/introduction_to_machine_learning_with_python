import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.savefig("wave_dataset.png")
print("X.shape: {}".format(X.shape)) # 40 samples, 1 feature each
print("y.shape: {}".format(y.shape))
print("X: \n{}".format(X))
print("y: \n{}".format(y))
