import numpy as np


# General function to compute expected hitting time for Exercise 1
def compute_Phi_ET(P, ns=100):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        ns {int} -- largest step to consider

    Returns:
        Phi_list {numpy.array} -- (ns + 1) x n x n, the Phi matrix for time 0, 1, ...,ns
        ET {numpy.array} -- n x n, expected hitting time approximated by ns steps ns
    '''

    # Add code here to compute following quantities:
    # Phi_list[m, i, j] = phi_{i,j}^{(m)} = Pr( T_{i, j} <= m )
    # ET[i, j] = E[ T_{i, j} ] ~ \sum_{m=1}^ns m Pr( T_{i, j} = m )
    # Notice in python the index starts from 0

    Phi_list = []

    # initialize list with phi_0
    phi_0 = [[0 for _ in range(len(P))] for _ in range(len(P))]
    for i in range(len(P)):
        for j in range(len(P)):
            phi_0[i][j] = 1 if i==j else 0
    Phi_list.append(np.array(phi_0))

    # recursively build rest of phi_list
    for m in range(1,ns+1):
        curr_phi = np.matmul(P,Phi_list[m-1]) # phi_m = delta_{ij} + (1-delta_{ij})P*phi_{m-1}
        for i in range(len(P)): # delta_{ij} = 1 if i=j, phi_m = 1 + 0(P*phi_{m-1}) = 1
            curr_phi[i][i] = 1               
        Phi_list.append(curr_phi)
    
    #calculate ET_ij = sum of mP(m) where Px = prob(T_ij = m)
    ET = [[0 for _ in range(len(P))] for _ in range(len(P))]
    for i in range(len(P)):
        for j in range(len(P)):
            for m in range(1,ns):
                ET[i][j] += m*((Phi_list[m]-Phi_list[m-1])[i][j])

    return Phi_list, ET


# General function to simulate hitting time for Exercise 1
def simulate_hitting_time(P, states, nr):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        states {list[int]} -- the list [start state, end state], index starts from 0
        nr {int} -- largest step to consider

    Returns:
        T {list[int]} -- a size nr list contains the hitting time of all realizations
    '''

    # Add code here to simulate following quantities:
    # T[i] = hitting time of the i-th run (i.e., realization) of process
    # Notice in python the index starts from 0
    T = []   
    for _ in range(nr):
        steps = 0
        curr = states[0]
        end = states[1]
        while curr != end and steps < nr:
            x = np.random.random()
            for j in range(len(P[curr])):
                x -= P[curr][j]
                if x <= 0:
                    curr = j
                    break
            steps += 1
        T.append(steps)
    return T



# General function to compute the stationary distribution (if unique) of a Markov chain for Exercise 3
def stationary_distribution(P):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain

    Returns:
        pi {numpy.array} -- length n, stationary distribution of the Markov chain
    '''

    # Add code here: Think of pi as column vector, solve linear equations:
    #     P^T pi = pi
    #     sum(pi) = 1

    from scipy.linalg import null_space

    I = np.identity(len(P))
    I_P_t = np.transpose(I-P)
    pi = null_space(I_P_t)
    pi = pi / np.sum(pi) # normalize

    return pi

