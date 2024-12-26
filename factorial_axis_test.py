import numpy as np

var1 = np.random.randint(1, 100, (10))
var2 = np.random.randint(1, 100, (10))
var3 = np.random.randint(1, 100, (10))

var1 = (var1 - np.mean(var1)) / np.std(var1)
var2 = (var2 - np.mean(var2)) / np.std(var2)
var3 = (var3 - np.mean(var3)) / np.std(var3)


vec = np.random.randint(1, 100, (3))
norm = np.sqrt(np.dot(vec, vec))

print('vec lenght = ', norm)
vec = vec / norm
print('after dividing by norm:', np.sqrt(np.dot(vec, vec)))


F = var1 * vec[0] + var2 * vec[1] + var3 * vec[2]


print('F variance test: ', np.sum(F**2) / 10)
print('F variance test: ', np.var(F))