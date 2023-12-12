import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

## Helper functions you don't need to modify

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

## End of helper functions

## Coding Exercise Starts Here

# General function to compute Vandermonde matrix for Exercise 2.2
def create_vandermonde(x, m):
    """
    Arguments:
        x {numpy.ndarray} -- 1d-array of (x_1, x_2, ..., x_n)
        m {int} -- a non-negative integer, degree of polynomial fit
    Returns:
        A {numpy.ndarray} -- an n x (m+1) matrix where A_{ij} = x_i^{j-1}
    """
    # Add code to compute Vandermonde A

    n = len(x)
    A = np.ones((n,m+1))
    for r in range(n):
        for i in range(1,m+1):
            A[r][i] = np.power(x[r],i)
    return A

# General function to solve linear least-squares via normal equations for Exercise 2.2
def solve_linear_LS(A, y):
    """
    Arguments:
        A {numpy.ndarray} -- an m x n matrix
        y {numpy.ndarray} -- a length-m vector
    Returns:
        z_hat {numpy.ndarray} -- length-n vector, the optimal solution for the given linear least-square problem
    """
    # Add code to compute least squares solution z_hat via linear algebra
    z_hat = np.linalg.lstsq(A,y, rcond=-1)[0]
    return z_hat

# General function to solve linear least-squares via via partial gradient descent for Exercise 2.2
def solve_linear_LS_gd(A, y, step, niter):
    """
    Arguments:
        A {numpy.ndarray} -- an m x n matrix
        y {numpy.ndarray} -- a length-m vector
        step -- a floating point number, step size
        niter -- a non-negative integer, number of updates
    Returns:
        z_hat {numpy.ndarray} -- length-n vector, the optimal solution for the given linear least-square problem
    """
    # Add code to approximate least squares solution z_hat via gradient descent
    z_hat = np.zeros(A.shape[1])
    N = A.shape[0]
    for i in range(niter):
        grad = (y[(i % N)] - (A[(i % N)] @ z_hat)) * A[(i % N)]
        z_hat = z_hat + step*grad
        
    return z_hat

# General function to extract samples with given labels and randomly split into test and training sets for Exercise 2.3
def extract_and_split(df, d, test_size=0.5):
    """
    extract the samples with given labels and randomly separate the samples into training and testing groups, extend each vector to length 785 by appending a −1
    Arguments:
        df {dataframe} -- the dataframe of MNIST dataset
        d {int} -- digit needs to be extracted, can be 0, 1, ..., 9
        test_size {float} -- the fraction of testing set, default value is 0.5
    Returns:
        X_tr {numpy.ndarray} -- training set features, a matrix with 785 columns
                                each row corresponds the feature of a sample
        y_tr {numpy.ndarray} -- training set labels, 1d-array
                                each element corresponds the label of a sample
        X_te {numpy.ndarray} -- testing set features, a matrix with 785 columns 
                                each row corresponds the feature of a sample
        y_te {numpy.ndarray} -- testing set labels, 1d-array
                                each element corresponds the label of a sample
    """
    # Add code here extract data and randomize order
    # return X_tr, X_te, y_tr, y_te  
    extracted_samples = df[df["label"]==d]
    X_tr, X_te, y_tr, y_te = train_test_split(extracted_samples["feature"], extracted_samples["label"], test_size=test_size, random_state=0)
    X_tr = X_tr.apply(lambda row: np.append(row,-1))
    X_te = X_te.apply(lambda row: np.append(row,-1))
    X_tr = np.vstack(X_tr)
    X_te = np.vstack(X_te)
    y_tr = np.array(y_tr)
    y_te = np.array(y_te) 
    return X_tr, X_te, y_tr, y_te 

# General function to train and test pairwise classifier for MNIST digits for Exercise 3.2
def mnist_pairwise_LS(df, a, b, test_size=0.5, verbose=False, gd=False):
    """
    Pairwise experiment for applying least-square to classify digit a and digit b
    Arguments:
        df {dataframe} -- the dataframe of MNIST dataset
        a, b {int} -- digits to be classified
        test_size {float} -- the fraction of testing set, default value is 0.5
        verbose {bool} -- whether to print and plot results
        gd {bool} -- whether to use gradient descent to solve LS        
    Returns:
        res {numpy.ndarray} -- numpy.array([training error, testing error])
    """
    # # Find all samples labeled with digit a and split into train/test sets
    Xa_tr, Xa_te, ya_tr, ya_te = extract_and_split(df, a, test_size=test_size)  

    # replace a label w -1
    ya_tr = np.full(len(ya_tr), -1) 
    ya_te = np.full(len(ya_te), -1)

    # # Find all samples labeled with digit b and split into train/test sets
    Xb_tr, Xb_te, yb_tr, yb_te = extract_and_split(df, b, test_size=test_size)

    # replace b label w 1
    yb_tr = np.full(len(yb_tr), 1) 
    yb_te = np.full(len(yb_te), 1)

    # # Construct the full training set
    X_tr = np.vstack((Xa_tr, Xb_tr))
    y_tr = np.append(ya_tr, yb_tr)

    # # Construct the full testing set
    X_te = np.vstack((Xa_te, Xb_te))
    y_te = np.append(ya_te, yb_te)    
    
    # # Run least-square on training set
    z_hat = np.linalg.lstsq(X_tr,y_tr, rcond=None)[0]

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
