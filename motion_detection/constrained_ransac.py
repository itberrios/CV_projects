"""
Constrained RANSAC Algorithm (CRA)

The contrained implies a Quasi-Random Sampling strategy to ensure an even coverage 
of the image is obtained rather than using Uniform Sampling which might not obtain
full coverage of the image


NOTE: 11/7/2023
    Need certain versions of numpy and llvmlite to support numba
    pip install numpy==1.23.5
    pip install llvmlite==0.40

"""

import numpy as np
from numba import jit


def get_sampling_index(w, h, s=50, p=0.5):
    """ Obtains sampling index along which to sample from
        Inputs:
            w - flow width
            h - flow height
            s - edge length to break image into squares
            P - portion of squares to sample 
        Outputs:
            index - (:,2) sampling index along x and y dimensions
            n_ttl - total possible number of points
            n_s  - actual number of points to sample
    """
    x_index, y_index = np.meshgrid(np.linspace(0, w - 1, w//s).astype(int), # x index
                                   np.linspace(0, h - 1, h//s).astype(int)) # y index

    n_ttl = x_index.shape[0]*y_index.shape[1] # total number of points to sample
    n_s = int(n_ttl*p) # number of points actually sampled

    x_index, y_index = x_index.reshape((-1, 1)), y_index.reshape((-1, 1))

    index = np.hstack((x_index, y_index))

    return index, n_ttl, n_s


def get_px(w, h):
    """ Obtains matrix of all points P 
        polynomial matrix X
        Inputs:
            w - flow width
            h - flow height
        Outputs:
            P - (N*M, 2) matrixof all data points
            X - (N*M, 6) matrix polynomial [x**2, y**2, x*y, x, y, 1] 
    """
    # get (x, y) coordinates for all points
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
    x, y = x.reshape((-1)), y.reshape((-1))

    # get matrix of all points
    P = np.vstack((x, y)).T

    # get X polynomial for all points
    X = np.array([x**2, y**2, x*y, x, y, np.ones((len(x),))]).T

    return P, X


jit(nopython=True) # , parallel=True, fastmath=True)
def cra(flow, P, X, index, n_ttl, n_s, thresh=0.01, min_inliers=10000, num_iters=50):
    """ Performs Constrained RANSAC to obtain background flow mask
        Inputs:
            flow - (N,M,2) Optical Flow Estimate
            P - (N*M, 2) matrixof all data points
            X - (N*M, 6) matrix polynomial [x**2, y**2, x*y, x, y, 1]
            index - (:,2) sampling index along x and y dimensions
            n_ttl - total possible number of points
            n_s  - actual number of points to sample
            thresh - RANSAC threshold to consider a point an inlier
            min_inliers - min number of inliers to consider a model
            num_iters - number of RANSAC iterations
        Outputs:
            H - best fit model (6x3) matrix
            best_error - error from best fit model
    """
    best_error = 1e10
    H = np.zeros((6, 2)) # 2
    
    # get reshaped mixed Flow
    F = flow.reshape((-1, 2))

    # initialize matrices 
    Ps = np.zeros((n_s, 2), dtype=int)
    Xs = np.zeros((n_s, 6))
    # Ys = np.zeros((n_s, 3))
    Ys = np.zeros((n_s, 2)) # 2
    Fb = np.zeros_like(F)

    for _ in range(num_iters):

        ## get (x, y) sample points (Ps)
        Ps[:, :] = index[np.sort(np.random.choice(np.arange(0, n_ttl), size=n_s , replace=False))]
        xs = Ps[:, 0]
        ys = Ps[:, 1]

        # get (u, v) samples
        Fs = flow[ys, xs, :]

        # get Y projected samples [x + u, y + v]^T
        # Ys[:, :] = np.hstack((Ps + Fs, np.ones((n_s, 1))))
        Ys[:, :] = Ps + Fs # 2
            
        # get X polynomial from samples [x^2, y^2, xy, x, y, 1]
        Xs[:, :] = np.array([xs**2, ys**2, xs*ys, xs, ys, np.ones((n_s,))]).T

        # Compute sample H matrix (fitting step)
        Hs = np.linalg.inv(Xs.T @ Xs) @ (Xs.T @ Ys)

        # Get background flow prediction using all points
        # Fb[:, :] = (X @ Hs)[:, :2] - P
        Fb[:, :] = (X @ Hs) - P # 2

        # get inlier locations
        inliers = np.mean(np.square(Fb - F), axis=1) < thresh

        # if there are enough inliers refit model to inliers
        if inliers.sum() > min_inliers:
            # get inlier X polynomial and Y projected
            Xi = X[inliers]
            # Yi = np.hstack((P[inliers] + F[inliers], np.ones((inliers.sum(), 1))))
            Yi = P[inliers] + F[inliers] # 2

            # fit H from inliers (X and Xi have swapped dimensions compared to Xs)
            Hi = np.linalg.inv(Xi.T @ Xi) @ (Xi.T @ Yi)

            # get new background estimate from inlier fit
            # Fi = (Xi @ Hi)[:, :2] - P[inliers] 
            Fi = (Xi @ Hi) - P[inliers] # 2

            # get error metric
            mse_i = np.mean(np.square(Fi - F[inliers, :]), axis=1)

            # if current error is better than previous error, save model (H matrix) and error
            if mse_i.sum() < best_error:
                best_error = mse_i.sum()
                H = Hi

    return H, best_error



jit(nopython=True) # , parallel=True, fastmath=True)
def cra_fast(flow, P, X, index, n_ttl, n_s, thresh=0.01, min_inliers=10000, num_iters=50):
    """ Performs Constrained RANSAC to obtain background flow mask. Attemps to be faster at the
        expense of less explainable code
        Inputs:
            flow - (N,M,2) Optical Flow Estimate
            P - (N*M, 2) matrixof all data points
            X - (N*M, 6) matrix polynomial [x**2, y**2, x*y, x, y, 1]
            index - (:,2) sampling index along x and y dimensions
            n_ttl - total possible number of points
            n_s  - actual number of points to sample
            thresh - RANSAC threshold to consider a point an inlier
            min_inliers - min number of inliers to consider a model
            num_iters - number of RANSAC iterations
        Outputs:
            H - best fit model (6x3) matrix
            best_error - error from best fit model
    """

    best_error = 1e10
    H = np.zeros((6, 2)) # 2
    
    # get reshaped mixed Flow
    F = flow.reshape((-1, 2))

    # initialize matrices 
    Ps = np.zeros((n_s, 2), dtype=int)
    Xs = np.zeros((n_s, 6))
    Ys = np.zeros((n_s, 2)) 
    Fb = np.zeros_like(F)

    

    for _ in range(num_iters):

        ## get (x, y) sample points (Ps)
        Ps[:, :] = index[np.sort(np.random.choice(np.arange(0, n_ttl), size=n_s , replace=False))]
        xs = Ps[:, 0]
        ys = Ps[:, 1]

        # get Y projected samples [x + u, y + v]^T
        # Ys[:, :] = Ps + flow[ys, xs, :]
            
        # get X polynomial from samples [x^2, y^2, xy, x, y, 1]
        Xs[:, :] = np.array([xs**2, ys**2, xs*ys, xs, ys, np.ones((n_s,))]).T

        # Compute sample H matrix (fitting step)
        # Hs = np.linalg.inv(Xs.T @ Xs) @ (Xs.T @ Ys)
        # Hs = np.linalg.inv(Xs.T @ Xs) @ (Xs.T @ (Ps + flow[ys, xs, :])) # faster

        # Get background flow prediction using all points
        # Fb[:, :] = (X @ Hs) - P
        Fb[:, :] = (X @ (np.linalg.inv(Xs.T @ Xs) @ (Xs.T @ (Ps + flow[ys, xs, :])))) - P # even faster??

        # get inlier locations
        inliers = np.mean(np.square(Fb - F), axis=1) < thresh

        # if there are enough inliers refit model to inliers
        if inliers.sum() > min_inliers:
            # get inlier X polynomial and Y projected
            Xi = X[inliers]
            # Yi = P[inliers] + F[inliers] 

            # fit H from inliers (X and Xi have swapped dimensions compared to Xs)
            # Hi = np.linalg.inv(Xi.T @ Xi) @ (Xi.T @ Yi)
            Hi = np.linalg.inv(Xi.T @ Xi) @ (Xi.T @ (P[inliers] + F[inliers])) # faster

            # get new background estimate from inlier fit
            Fi = (Xi @ Hi) - P[inliers] 

            # get error metric
            mse_i = np.mean(np.square(Fi - F[inliers, :]), axis=1)

            # if current error is better than previous error, save model (H matrix) and error
            if mse_i.sum() <  best_error:
                best_error = mse_i.sum()
                H = Hi

    return H, best_error