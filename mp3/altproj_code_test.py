import numpy as np
import pandas as pd
from altproj_code import altproj, kaczmarz, lp_altproj

#
# Desired output of this script
# -----------------------------
#
# altproj Example 1 err:
#  [1.0752     0.99090432 0.91321742 0.84162118 0.77563808]
# altproj Example 1:v
#  [0.58172856 0.77563808] 
#
# altproj Example 2 err:
#  3.3587408979332825
# altproj Example 2 v:
#  [5.80196883 5.7414081  5.80196883 4.7037591  5.80196883] 
#
# altproj Example 3 err:
#  0.020871922644255747
# altproj Example 3 v:
#  [2.70685396 5.37727694 2.70685396 8.04162808 2.70685396] 
#
# kaczmarz Example 4 err:
#  0.028524403942799426
# kaczmarz Example 4 X:
#  [-0.13512375  0.0941078  -0.13512375  0.39621042 -0.13512375] 
#
# kaczmarz Example 5 err:
#  0.01468037281401613
# kaczmarz Example 5 X:
#  [-0.03184633 -0.03805198  0.04546457  0.02235128  0.04669109  0.108116  ] 
#
#

### If you project on x >= d 1 constraints at the end of each pass
# lp_altproj Example 6 err:
#  0.05188146770730828
# lp_altproj Example 6 v:
#  [-1.96541235 -1.98270618] 
#
# lp_altproj Example 7 err:
#  0.7250551085496868
# lp_altproj Example 7 v:
#  [2.01660003 1.81980856 1.62730644] 

### If you automatically project on the x >= d 1 constraints after each row of A
# lp_altproj Example 6 err:
#  3.333333318673178
# lp_altproj Example 6 v:
#  [0.66666666 0.        ] 
#
# lp_altproj Example 7 err:
#  0.6566929819164207
# lp_altproj Example 7 v:
#  [2.02850424 1.83679799 1.64897667] 


# Test altproj
A = np.array([[4, 3]]).T
B = np.array([[3, 4]]).T
v0 = np.array([1, 1])
n = 5
v, err = altproj(A, B, v0, n)
print("altproj Example 1 err:\n",err)
print("altproj Example 1:v\n",v,"\n")

A = np.array([[1, 2, 1, 3, 1],[6, 7, 6, 7, 6]]).T
B = np.array([[1, 2, 1, 3, 1],[7, 8, 7, 8, 7]]).T
v0 = np.array([10, 2, 10, 3, 10])
# n = 5
n = 5000
v, err = altproj(A, B, v0, n)
print("altproj Example 2 err:\n",err[-1])
print("altproj Example 2 v:\n",v,"\n")
n = 40000
v, err = altproj(A, B, v0, n)
print("altproj Example 3 err:\n",err[-1])
print("altproj Example 3 v:\n",v,"\n")

# Test kaczmarz
A = np.array([[1, 2, 1, 3, 1],[6, 7, 6, 7, 6]])
b = np.array([1, 1])
X, err = kaczmarz(A, b, 20)
print("kaczmarz Example 4 err:\n",err[-1])
print("kaczmarz Example 4 X:\n",X[:,-1],"\n")

A = np.array([[4, 2, 7, 3, 1, 7],[6, 7, 6, 7, 6, 7],[1, 2, 3, 4, 5, 6]])
b = np.array([1, 1, 1])
X, err = kaczmarz(A, b, 20)
print("kaczmarz Example 5 err:\n",err[-1])
print("kaczmarz Example 5 X:\n",X[:,-1],"\n")

# Test lp_altproj
A = np.array([[-2, 1],[1, -2]])
b = np.array([2,2])
v, err = lp_altproj(A, b, 10, d=-np.Inf)
print("lp_altproj Example 6 err:\n",err[-1])
print("lp_altproj Example 6 v:\n",v,"\n")

A = np.array([[-4, 3,3],[3, -4, 3],[3, 3, -4]])
b = np.array([3,4,5])
v, err = lp_altproj(A, b, 10, d=-np.Inf)
print("lp_altproj Example 7 err:\n",err[-1])
print("lp_altproj Example 7 v:\n",v,"\n")