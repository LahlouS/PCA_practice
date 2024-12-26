import numpy as np
import pandas as pd

X = pd.read_csv('./data.csv')

n, p = len(X), len(X.columns)
print(n, p)

Z = (X - np.mean(X, axis=0))/ np.std(X, axis=0)

ZZT = np.matmul(Z, Z.T)
ZTZ = np.matmul(Z.T, Z)

print("ZZT.shape:", ZZT.shape)
print("ZTZ.shape:", ZTZ.shape)

print("determinant of ZZT: ", np.linalg.det(ZZT))
print("determinant of ZTZ: ", np.linalg.det(ZTZ))

print("rank of ZZ.T", np.linalg.matrix_rank(ZZT))
print("rank of Z.TZ", np.linalg.matrix_rank(ZTZ))