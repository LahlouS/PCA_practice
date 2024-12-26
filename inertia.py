import numpy as np
import pandas as pd

print('TEST 1')
print()
mat = np.random.randint(500, 1200, (40, 12))

ind_mu = [np.mean(mat[:, i]) for i in range(12)]
ind_var = [np.var(mat[:, i]) for i in range(12)]

var_mu = [np.mean(mat[i, :]) for i in range(40)]
var_var = [np.var(mat[i, :]) for i in range(40)]

ind_inertia = np.sum( [(np.sum(((mat[i, :] - ind_mu)**2) / 40) ) for i in range(40)])

var_inertia = np.sum( [(np.sum(((mat[:, i] - var_mu)**2) / 12)) for i in range(12)])

print(var_inertia, "<==>", ind_inertia)

try:
    assert np.isclose(var_inertia, ind_inertia), 'TEST 1: both inertia should be the same'
except AssertionError as e:
    print(e) 

print('TEST 2')
print('')

centered_ind = mat - mat.mean()
centered_var = mat.T - mat.T.mean()

n = 40
p = 12

inertia_ind = np.sum((np.linalg.norm(centered_ind, axis=1) ** 2) / n)
inertia_var = np.sum((np.linalg.norm(centered_var, axis=1) ** 2) / p)

print(f"Inertia from individuals' perspective (individual cloud): {inertia_ind}")
print(f"Inertia from variables' perspective (variable cloud): {inertia_var}")

try:
    assert np.isclose(var_inertia, ind_inertia), 'TEST 2: both inertia should be the same'
except AssertionError as e:
    print(e) 

print('TEST 3')
print()

# Generate a random dataset for demonstration
np.random.seed(0)
df = pd.DataFrame(np.random.rand(10, 5), columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5'])

# 1. Standardization (manual z-score normalization)
# Compute the mean and standard deviation for each variable (column)
means = df.mean(axis=0)  # Mean for each column (variable)
std_devs = df.std(axis=0)  # Standard deviation for each column (variable)

# Standardize the data: (value - mean) / std_dev
df_standardized = (df - means) / std_devs

# 2. Inertia of the individual cloud (rows as points)
# Centered and standardized data for individuals: each individual is a point in variable space
n = df_standardized.shape[0]  # Number of individuals
centered_individuals = df_standardized  # Data is already standardized and centered
inertia_individuals = np.sum(np.linalg.norm(centered_individuals, axis=1) ** 2) / n

# 3. Inertia of the variable cloud (columns as points)
# Centered and standardized data for variables: each variable is a point in individual space
p = df_standardized.shape[1]  # Number of variables
centered_variables = df_standardized.T  # Transpose the data
inertia_variables = np.sum(np.linalg.norm(centered_variables, axis=1) ** 2) / p

# Output the results
print(f"Inertia from individuals' perspective (individual cloud): {inertia_individuals}")
print(f"Inertia from variables' perspective (variable cloud): {inertia_variables}")

try:
    assert np.isclose(var_inertia, ind_inertia), 'TEST 3: both inertia should be the same'
except AssertionError as e:
    print(e) 


print('TEST 4')
print()


# Generate a random dataset for demonstration
np.random.seed(0)
df = pd.DataFrame(np.random.rand(10, 5), columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5'])

# 1. Centering the data (without standardizing)
# Subtract the mean of each variable (column-wise centering)
centered_df = df - df.mean(axis=0)

# 2. Inertia of the individual cloud (rows as points in variable space)
# Centered data for individuals: each individual is a point in variable space
n = centered_df.shape[0]  # Number of individuals
# Inertia is the sum of the squared distances to the centroid (centroid is 0 after centering)
inertia_individuals = np.sum(np.linalg.norm(centered_df, axis=1) ** 2) / n

# 3. Inertia of the variable cloud (columns as points in individual space)
# Centered data for variables: each variable is a point in individual space
p = centered_df.shape[1]  # Number of variables
# We transpose the data to treat variables as points in the individual space
centered_df_transposed = centered_df.T
# Compute the inertia for variables
inertia_variables = np.sum(np.linalg.norm(centered_df_transposed, axis=1) ** 2) / p

# Output the results
print(f"Inertia from individuals' perspective (individual cloud): {inertia_individuals}")
print(f"Inertia from variables' perspective (variable cloud): {inertia_variables}")

try:
    assert np.isclose(inertia_individuals, inertia_variables), "Inertia should be the same!"
except AssertionError as e:
    print(e)

print("TEST 5")

np.random.seed(0)
X = np.random.randint(50, 100, (10, 5))
df = pd.DataFrame(X, columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5'])
print(df.head())
# standardising our variable
means = np.mean(X, axis=0)
stds = np.std(X, axis = 0)

Z = X - means
Z = Z / stds
print("************STD DATA**************")
print(Z)
print("******************************************")


var1 = np.sum(Z**2, axis=1) / 10
var2 = np.sum(Z**2, axis=0) / 10
print('individuals cloud of point variance = ', var1, ' sum --> ', np.sum(var1))
print('variables cloud of point variance = ', var2, 'sum --> ', np.sum(var2))
print()