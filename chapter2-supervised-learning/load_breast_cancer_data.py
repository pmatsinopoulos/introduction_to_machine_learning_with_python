from sklearn.datasets import load_breast_cancer
import numpy as np


cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("Shape of cancer data: {}".format(cancer.data.shape))
print("Number of samples: {}".format(cancer.data.shape[0]))
print("Number of features: {}".format(cancer.data.shape[1]))
print("Number of feature names: {}".format(len(cancer.feature_names)))
print("Feature names: \n{}".format(cancer.feature_names))
print("Sample counts per class: \n{}".format(
    {str(n): int(v) for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
))
print("Cancer description: \n{}".format(cancer.DESCR))
