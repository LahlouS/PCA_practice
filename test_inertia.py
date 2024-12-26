import numpy as np

n, p = 10, 5

X = np.random.randint(50, 100, (n, p)).astype(float)
print('X = ', X)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

Z = (X - mean) * std

res1 = np.dot(Z.T, Z)

print('np.dot(X.T, X): \n', res1.shape)

res2 = [np.outer(Z[i], Z[i]) for i in range(n)]
res2 = np.array(res2)
print('before summation', res2.shape)
res2 = np.sum(res2, axis=0)
print('after summation:', res2.shape)

print('******* FINAL RESULT *********')

print('np.dot(X.T, X): \n', np.round(res1, decimals=3))

print()
print('sum(x_i, x_i.T): \n', np.round(res2, decimals=3))