import numpy as np
import pandas as pd

n = 40
mecannic = np.random.randint(1000, 1200, (n))
dr = np.random.randint(2000, 3000, (n))
waitress = np.random.randint(500, 700, (n))


DS = pd.DataFrame(data={
    'mec': mecannic,
    'dr': dr,
    'wait': waitress
})
X = DS.values

X_centered = X - np.mean(X, axis=0)
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
assert np.all(X_norm[:, 0] == ((mecannic - np.mean(mecannic)) / np.std(mecannic))), 'ERREUR AXES DE STANDARDISATION'

cov_matrix = np.dot(X_centered.T, X_centered)
correlation_matrix = np.dot(X_norm.T, X_norm) * 1/n

res = np.zeros((3, 3))
print(res)
for i in range(n):
    res += np.outer(X_norm[i], X_norm[i])

res *= 1/n

for ir in range(3):
    print(res[ir], '\t\t', correlation_matrix[ir])