import numpy as np
import pandas as pd
from lsq_code import create_vandermonde, solve_linear_LS, solve_linear_LS_gd, mnist_pairwise_LS

#
# Desired output of this script
# -----------------------------
#
# Vandermonde Example 1:
#  [[1 1 1]
#  [1 3 9]
#  [1 2 4]] 

# Vandermonde Example 2:
#  [[ 1 -2  4 -8]
#  [ 1 -1  1 -1]
#  [ 1  0  0  0]
#  [ 1  1  1  1]
#  [ 1  2  4  8]] 

# solve_linear_LS Example 1:
#  [-4.   6.5 -1.5] 

# solve_linear_LS Example 2:
#  [ 1.25714286  0.58333333  0.07142857 -0.08333333] 

# solve_linear_gd Example 1:
#  [-3.92367303  6.41369966 -1.47965981] 

# solve_linear_gd Example 2:
#  [ 1.24030161  0.51307179  0.07121913 -0.06784267] 

# mnist_pairwise_LS Example 0 :
# Pairwise experiment, mapping 2 to -1, mapping 3 to 1
# training error = 1.85%, testing error = 4.20%
# Confusion matrix:
#  [[2013   75]
#  [ 104 2071]]
# results:  [0.01852286 0.04198921] 

# mnist_pairwise_LS Example 1 :
# Pairwise experiment, mapping 2 to -1, mapping 3 to 1
# training error = 1.85%, testing error = 3.85%
# Confusion matrix:
#  [[2015   73]
#  [  91 2084]]
# results:  [0.01852286 0.03847056] 

# mnist_pairwise_LS Example 2 :
# Pairwise experiment, mapping 2 to -1, mapping 3 to 1
# training error = 1.83%, testing error = 3.94%
# Confusion matrix:
#  [[2002   86]
#  [  82 2093]]
# results:  [0.01828839 0.03940887] 


A1 = create_vandermonde(np.asarray([1,3,2]),2)
print("Vandermonde Example 1:\n",A1,"\n")

A2 = create_vandermonde(np.arange(-2,3),3)
print("Vandermonde Example 2:\n",A2,"\n")

z1 = solve_linear_LS(A1,np.asarray([1,2,3]))
print("solve_linear_LS Example 1:\n",z1,"\n")

z2 = solve_linear_LS(A2,np.asarray([1,1,1,2,2]))
print("solve_linear_LS Example 2:\n",z2,"\n")

z1 = solve_linear_LS_gd(A1,np.asarray([1,2,3]),0.01,50000)
print("solve_linear_gd Example 1:\n",z1,"\n")

z2 = solve_linear_LS_gd(A2,np.asarray([1,1,1,2,2]),0.01,1000)
print("solve_linear_gd Example 2:\n",z2,"\n")

df = pd.read_csv('mnist_train.csv')
df['feature'] = df.apply(lambda row: row.values[1:], axis=1)
df = df[['feature', 'label']]
for i in range(3):
    print("mnist_pairwise_LS Example",i,":")
    res = mnist_pairwise_LS(df, 2, 3, verbose=True)
    print("results: ",res,"\n")

