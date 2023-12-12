import numpy as np
#from numba import njit
from scipy.linalg import inv, svd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from lsq_code import extract_and_split
from sklearn.model_selection import train_test_split


### Helper functions

# Function to remove outliers before plotting histogram
def remove_outlier(x, thresh=3.5):
    """
    returns points that are not outliers to make histogram prettier
    reference: https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting/11886564
    Arguments:
        x {numpy.ndarray} -- 1d-array, points to be filtered
        thresh {float} -- the modified z-score to use as a threshold. Observations with
                          a modified z-score (based on the median absolute deviation) greater
                          than this value will be classified as outliers.
    Returns:
        x_filtered {numpy.ndarray} -- 1d-array, filtered points after dropping outlier
    """
    if len(x.shape) == 1: x = x[:,None]
    median = np.median(x, axis=0)
    diff = np.sqrt(((x - median)**2).sum(axis=-1))
    modified_z_score = 0.6745 * diff / np.median(diff)
    x_filtered = x[modified_z_score <= thresh]
    return x_filtered

# Compute null space
def null_space(A, rcond=None):
    """
    Compute null spavce of matrix XProjection on half space defined by {v| <v,w> = c}
    Arguments:
        A {numpy.ndarray} -- matrix whose null space is desired
        rcond {float} -- intercept
    Returns:
        Q {numpy.ndarray} -- matrix whose (rows?) span null space of A
    """
    u, s, vh = svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

### End Helper Functions
# Exercise 1: Alternating projection for subspaces
def altproj(A, B, v0, n):
    """
    Arguments:
        A {numpy.ndarray} -- matrix whose columns form basis for subspace U
        B {numpy.ndarray} -- matrix whose columns form basis for subspace W
        v0 {numpy.ndarray} -- initialization vector
        n {int} -- number of sweeps for alternating projection
    Returns:
        v {numpy.ndarray} -- the output after 2n steps of alternating projection
        err {numpy.ndarray} -- the error after each full pass
    """
    
    ### Add code here
    
    basis_UintW = np.hstack([A, B]) @ null_space(np.hstack([A, -B]))     

    actual = (basis_UintW@np.linalg.pinv(basis_UintW))@v0

    PA = A@np.linalg.pinv(A)
    PB = B@np.linalg.pinv(B)

    v = v0
    err = []
    for i in range(2*n):
        if i % 2 == 0: #even
            v = PA@v
            if i != 0:
                err.append(np.linalg.norm(v-actual, np.inf))
        else: #odd
            v = PB@v

    err.append(np.linalg.norm(v-actual, np.inf))

    return v, err

# Exercise 2: Kaczmarz algorithm for solving linear systems
#@njit
def kaczmarz(A, b, I):
    """
    Arguments:
        A {numpy.ndarray} -- matrix defines the LHS of linear equation
        b {numpy.ndarray} -- vector defines the RHS of linear equation
        I {int} -- number of full passes through the Kaczmarz algorithm
    Returns:
        X {numpy.ndarray} -- the output of all I full passes
        err {numpy.ndarray} -- the error after each full pass
    """
    
    ### Add code here
    m, n = A.shape
    v = np.zeros(n)
    X = []
    err = []
    for _ in range(I):
        for i in range(m):
            a_sigma = A[i]
            vi_dot_a_sigma_i = np.dot(a_sigma, v)
            v = v - (vi_dot_a_sigma_i - b[i])/np.dot(a_sigma,a_sigma) * a_sigma
        X.append(v)
        err.append(np.linalg.norm(A@v-b, np.inf))

        
    return np.array(X).T, np.array(err)

# Exercise 4: Alternating projection to satisfy linear inequalities
#@njit
def lp_altproj(A, b, I, s=1, d=0):
    """
    Find a feasible solution for A v >= b using alternating projection
    starting from v0 = 0
    Arguments:
        A {numpy.ndarray} -- matrix defines the LHS of linear equation
        b {numpy.ndarray} -- vector defines the RHS of linear equation
        I {int} -- number of full passes through the alternating projection
        s {numpy.float} -- step size of projection (defaults to 1)
        d {numpy.float} -- coordinate lower bound for all elements of v (defaults to 0)
    Returns:
        v {numpy.ndarray} -- the output after I full passes
        err {numpy.ndarray} -- the error after each full pass
    """
    # Add code here
    m = A.shape[0]
    v = np.zeros(A.shape[1])
    err = []

    for _ in range(I):
        # alternating projection
        for i in range(m):
            innerProduct = np.dot(A[i],v)

            if innerProduct >= b[i]:
                proj = v
            else: #innerProduct < b[i]:
                proj = v - ((innerProduct-b[i])/np.dot(A[i],A[i]) * A[i])
            
            v = (1-s)*v + s*proj

        # project onto non negativeity constraints
        if d != None:
            for i in range(len(v)):
                v[i] = max(d,v[i])

        err.append(np.max(np.max(b-A@v), 0))
    
    return v, err

def mnist_pairwise_altproj(df, a, b, solver, verbose=False):
    # s small, I large

    Xa_tr, Xa_te, ya_tr, ya_te = extract_and_split(df, a, test_size=0.5)  
    Xb_tr, Xb_te, yb_tr, yb_te = extract_and_split(df, b, test_size=0.5)  

    # -1 = a
    ya_tr = np.full(len(ya_tr), -1) 
    ya_te = np.full(len(ya_te), -1)

    # 1 == b
    yb_tr = np.full(len(yb_tr), 1) 
    yb_te = np.full(len(yb_te), 1)


    # # Construct the full training set
    X_tr = np.vstack((Xa_tr, Xb_tr))
    y_tr = np.append(ya_tr, yb_tr)

    # # Construct the full testing set
    X_te = np.vstack((Xa_te, Xb_te))
    y_te = np.append(ya_te, yb_te)   

    # min x that Ax >= b=1
    A = np.vstack((Xa_tr*-1, Xb_tr))
    b1 = np.append(ya_tr*-1, yb_tr)

    # part5 
    z_hat, err = solver(A,b1)

    # # Compute estimate and classification error for training set
    y_hat_tr = np.sign(X_tr @ z_hat) 
    incorrect_tr = sum(y_tr != y_hat_tr) #if non zero, incorrect classification. zero is correct (-1 - -1 = 0, 1-1=0)
    err_tr = incorrect_tr/len(y_hat_tr)

    # # Compute estimate and classification error for testing set
    y_hat_te = np.sign(X_te @ z_hat)

    incorrect_te = sum(y_te != y_hat_te) #if non zero, correct classification (-1 - -1 = 0, 1-1=0)
    err_te = incorrect_te/len(y_hat_te)
    
    if verbose:
        print('Pairwise experiment, mapping {0} to -1, mapping {1} to 1'.format(a, b))
        print('training error = {0:.2f}%, testing error = {1:.2f}%'.format(100 * err_tr, 100 * err_te))
        
        # Compute confusion matrix for train
        cm_r = np.zeros((2, 2), dtype=np.int64)
        cm_r[0, 0] = ((y_tr == -1) & (y_hat_tr == -1)).sum()
        cm_r[0, 1] = ((y_tr == -1) & (y_hat_tr == 1)).sum()
        cm_r[1, 0] = ((y_tr == 1) & (y_hat_tr == -1)).sum()
        cm_r[1, 1] = ((y_tr == 1) & (y_hat_tr == 1)).sum()
        print('Training Confusion matrix:\n {0}'.format(cm_r))

        # Compute confusion matrix for test
        cm_e = np.zeros((2, 2), dtype=np.int64)
        cm_e[0, 0] = ((y_te == -1) & (y_hat_te == -1)).sum()
        cm_e[0, 1] = ((y_te == -1) & (y_hat_te == 1)).sum()
        cm_e[1, 0] = ((y_te == 1) & (y_hat_te == -1)).sum()
        cm_e[1, 1] = ((y_te == 1) & (y_hat_te == 1)).sum()
        print('Testing Confusion matrix:\n {0}'.format(cm_e))

        # Compute the histogram of the function output separately for each class 
        # Then plot the two histograms together

        ya_te_hat, yb_te_hat = Xa_te @ z_hat, Xb_te @ z_hat
        output = np.append(remove_outlier(ya_te_hat),remove_outlier(yb_te_hat))
        plt.figure(figsize=(8, 4))
        plt.hist(output, bins=50)
    
    res = np.array([err_tr, err_te])
    return res

def mnist_multiclass_altproj(df, solver):

    ### create train and test sets
    X_tr, X_te, y_tr, y_te = train_test_split(df["feature"], df["label"], test_size=0.5, random_state=0)
    # add -1 to each inp vector
    X_tr = X_tr.apply(lambda row: np.append(row,-1))
    X_te = X_te.apply(lambda row: np.append(row,-1))

    # convert from pd.series to np.arrays
    X_tr = np.vstack(X_tr)
    X_te = np.vstack(X_te)
    y_tr = np.array(y_tr)
    y_te = np.array(y_te) 
    numClasses = 10

    # construct A matrix
    A = []
    for xi, yi in tqdm(zip(X_tr, y_tr), total=y_tr.size, leave=False):
        Ai_tilde = -np.kron(np.eye(10), xi) +np.kron(np.eye(10)[yi], xi[None,:])
        A.append(Ai_tilde[np.arange(10) != yi])
    A = np.vstack(A)
    print("1completed A matrix")

    # construct b vector
    b = np.zeros(A.shape[0])

    # solve
    z_hat, err = solver(A,b)

    z_hat = z_hat.reshape((numClasses,-2))
    z_hat = np.transpose(z_hat)
    # # Compute estimate and classification error for training set
    y_hat_tr = X_tr @ z_hat
    # print(y_hat_tr.shape)
    y_hat_tr = np.argmax(y_hat_tr, axis=1)
    # print(y_hat_tr)

    incorrect_tr = sum(y_tr != y_hat_tr) #if non zero, incorrect classification. zero is correct (-1 - -1 = 0, 1-1=0)
    err_tr = incorrect_tr/len(y_hat_tr)

    # # Compute estimate and classification error for testing set
    y_hat_te = np.argmax(X_te @ z_hat,axis=1)

    incorrect_te = sum(y_te != y_hat_te) #if non zero, correct classification (-1 - -1 = 0, 1-1=0)
    err_te = incorrect_te/len(y_hat_te)
    
    print('training error = {0:.2f}%, testing error = {1:.2f}%'.format(100 * err_tr, 100 * err_te))
    # confusion matrices
    cm = np.zeros((numClasses, numClasses), dtype=np.int64)
    for a in range(numClasses):
        for b in range(numClasses):
            cm[a, b] = ((y_tr == a) & (y_hat_tr == b)).sum()
    print('Training Confusion matrix:\n {0}'.format(cm))

    cm = np.zeros((numClasses, numClasses), dtype=np.int64)
    for a in range(numClasses):
        for b in range(numClasses):
            cm[a, b] = ((y_te == a) & (y_hat_te == b)).sum()
    print('Testing Confusion matrix:\n {0}'.format(cm))
    
    res = np.array([err_tr, err_te])
    return z_hat, res