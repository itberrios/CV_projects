"""
    Implements Normalized Convolution

"""

import scipy
import numpy as np


def normalized_convolution(f: np.ndarray, c: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Computes Normalized Convolution and returns the coefficients r as well as 
        The polynomial basis functions in the x and y dimensions. For the efficieny,
        the computations are implemented as separable convolutions. The signal is 
        always given the following 2nd order polynomial basis: {1, x, y, x2, xy, y2}.

        Inputs: 
            f - signal
            c - signal certainty (same shape as the signal)
            a - (nxn) applicability kernel for each pixel

        Outputs
            r - Computed Coefficients --> parameterization for 2nd order polynomial
            bx - x-dimension basis functions
            by - y-dimension basis functions
            
        References:
            [1]    G. Farneback, “Polynomial Expansion for Orientation and Motion Estimation,” 
            Ph.D. dissertation, Inst. of  Technology, Linköping Univ., Norrköping, Sweden, 2002. 
            [Online]. Available: http://liu.diva-portal.org/smash/get/diva2:302485/FULLTEXT01.pdf
            (Chapter 3)
        """
    # get dimensionality of applicability
    n = a.shape[0]

    # reshape applicability kernel
    a = a.reshape(-1)
    
    # get Basis functions
    # split Basis Functions between x and y dimensions
    b = np.repeat(np.c_[np.arange(-(n-1)//2, (n+1)//2)], 
                  5, axis=-1).T

    # x dimension (set y=1)
    bx = np.vstack((
        np.ones((n**2)),    # 1
        b.T.reshape(-1),    # x
        np.ones((n**2)),    # y
        b.T.reshape(-1)**2, # x^2
        b.T.reshape(-1),    # xy
        np.ones((n**2)),    # y^2
    )).T

    # y dimension (set x=1)
    by = np.vstack((
        np.ones((n**2)),  # 1
        np.ones((n**2)),  # x
        b.reshape(-1),    # y
        np.ones((n**2)),  # x^2
        b.reshape(-1),    # xy
        b.reshape(-1)**2, # y^2
    )).T


    # Pre-calculate product of certainty and signal
    cf = c * f

    # G and v are used to calculate "r" from the paper: v = G*r
    # r is the parametrization of the 2nd order polynomial for f
    G = np.empty(list(f.shape) + [bx.shape[-1]] * 2)
    v = np.empty(list(f.shape) + [bx.shape[-1]])

    # Apply separable cross-correlations

    # Pre-calculate quantities recommended in paper
    ab_x = np.einsum("i,ij->ij", a, bx)
    abb_x = np.einsum("ij,ik->ijk", ab_x, bx)

    # Calculate G and v for each pixel with cross-correlation
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(
                c, abb_x[..., i, j], axis=0, mode="constant", cval=0
            )

        v[..., i] = scipy.ndimage.correlate1d(
            cf, ab_x[..., i], axis=0, mode="constant", cval=0
        )

    
    # Pre-calculate quantities recommended in paper
    ab_y = np.einsum("i,ij->ij", a, by)
    abb_y = np.einsum("ij,ik->ijk", ab_y, by)

    # Calculate G and v for each pixel with cross-correlation
    # Note that these are of the form x = x*b or x *= b
    # We have already included the c and cf terms in the x-dim computation
    for i in range(by.shape[-1]):
        for j in range(by.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(
                G[..., i, j], abb_y[..., i, j], axis=1, mode="constant", cval=0
            )

        v[..., i] = scipy.ndimage.correlate1d(
            v[..., i], ab_y[..., i], axis=1, mode="constant", cval=0
        )

    # if G is singular, then we will need to reduce the singularity 
    # if np.any(~np.isfinite(np.linalg.cond(G))):
    #     # add a small Normal perturbation to reduce singularity of G
    #     G += np.random.normal(loc=0, scale=0.001, size=G.shape)

    # Solve r for each pixel
    r = np.linalg.solve(G, v)

    return (r, bx, by)