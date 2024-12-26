import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

p = 6
n = 18
X = np.random.randint(100, 2000, (n, p))

means = np.mean(X, axis=0)
stds = np.std(X, axis=0, ddof=0)
print('mean per variable', means)
print('standart deviation per variable', stds, end='\n\n')

Z = (X - means) / stds

print('shape of Z', Z.shape, end='\n\n')

pca = PCA()
F = pca.fit_transform(Z)
print('shape of F', F.shape, end='\n\n')
cat = np.concatenate((F, Z), axis = 1)
cat = (cat - np.mean(cat, axis=0)) / np.std(cat, axis=0)



corr = np.dot(cat.T, cat) / n
print('corr.shape ->', corr.shape)
print('FULL MATRIX')
for i in range(12):
    for j in range(12):
        print(round(corr[i, j], 3), end='\t')
    print()

corr = corr[:p, p:]
print('corr.shape ->', corr.shape)
print('CORRELATION BETWEEN AXIS AND VARIABLES MATRIX')
for i in range(6):
    for j in range(6):
        print(round(corr[i, j], 3), end='\t')
    print()

print('FROM THE CORRELATION BETWEEN AXIS AND VARIABLES')
lamdb_k = np.sum(corr**2, axis=1)
print()
print('lamda_k:', lamdb_k)
print('lambda total:', np.sum(lamdb_k))

print('FROM PROJECTED VARIANCE ALONG EACH AXIS')
lamdb_k = np.var(F, axis=0)
lamdb_k_bis = np.sum(F**2, axis=0) / n
print()
print('lambda k:', lamdb_k)
print('lambda k bis:', lamdb_k_bis)
print('lambda total:', np.sum(lamdb_k))


R = np.dot(Z.T, Z) / n
print()
print('correlation between variable 1 and 2 using R:', R[0, 1])
print('approximated correlation between variable 1 and 2 all the F:', corr[0, 0] * corr[0, 1] 
                                                                                    + corr[1, 0] * corr[1, 1] 
                                                                                    + corr[2, 0] * corr[2, 1]
                                                                                    + corr[3, 0] * corr[3, 1]
                                                                                    + corr[4, 0] * corr[4, 1]
                                                                                    + corr[5, 0] * corr[5, 1])