import numpy as np

# extra functions for reference
def f0(r):
    """ DC center sinusoidal in the radius
        r - is the Radius from the center 
    """
    outputs = r.copy().astype(np.float32)

    index = (r > 56) & (r <= 256)

    outputs[r <= 56] = 127
    outputs[index] = 127*np.sin(r[index]/np.pi)
    outputs[r > 256] = 127

    return outputs


def f1(r, c=8e-4):
    """ 
        DC center, sinusoidal chirp up out from radius
        r - is the Radius from the center 
        c - controls the radial chirp rate
    """
    outputs = r.copy().astype(np.float32)

    index = (r > 28) & (r <= 256)

    outputs[r <= 28] = 127
    outputs[index] = 127*(1 + np.cos(r[index]/(4*np.pi) + (c * (r[index]**2))))
    outputs[r > 256] = 127

    return outputs



g = lambda r : np.sin(((112*np.pi)/(np.log(2))*((2**(-r/56))) - 2**(-256/56)))


def f(r):
    """ r - is the Radius from the center """
    outputs = r.copy()

    index_1 = (r > 56) & (r <= 64)
    index_2 = (r > 64) & (r <= 224)
    index_3 = (r > 224) & (r <= 256)

    outputs[r < 56] = 127
    outputs[index_1] = 127*(1 + (g(r[index_1]) * np.cos((np.pi*r[index_1]/16) - 4*np.pi)**2))
    outputs[index_2] = 127*(1 + g(r[index_2]))
    outputs[index_3] = 127*(1 + (g(r[index_3]) * np.sin((np.pi*r[index_3]/64) - 4*np.pi)**2))
    outputs[r > 256] = 127

    return outputs


def get_test_image(n, func=f):
    """ Obtains an NxN test image """

    # get image indexes as column vectors
    x_index, y_index = np.meshgrid(np.arange(0, n), np.arange(0, n))
    x_index = x_index.reshape((-1, 1))
    y_index = y_index.reshape((-1, 1))

    # get radius r
    r = np.round(np.sqrt((x_index - n//2)**2 + (y_index - n//2)**2)).astype(np.float32).squeeze()

    return func(r).reshape((n, n))