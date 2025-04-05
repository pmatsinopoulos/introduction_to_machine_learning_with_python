import pandas as pd
import numpy as np
import mglearn

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

print("Data shape: {}".format(data.shape)) # (506, 13)
print("Target shape: {}".format(target.shape)) # (506,)
print("First 5 rows of data:\n{}".format(data[:5]))

X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape)) # (506, 104)
