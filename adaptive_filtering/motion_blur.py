import numpy as np


def motion_transform(u, v, a, b, T):
    """ Obtains Motion transform H in Fourier Space
        Inputs:
            u - horizontal frequency (wave) number
            v - vertical frequency (wave) number
            a - horizontal motion component
            b - vertical motion component
            T - time interval for image capture
        Outputs:
            H - Motion Transformation in Fourier Space
        """
    if (a == 0) and (b == 0):
        return 1
    
    omega = np.pi*(u*a + v*b)
    H = (T/omega) * np.sin(omega) * np.exp(-(1j * omega))

    # remove NaN
    H[np.isnan(H)] = 1 + 1j

    # normlize H
    H /= np.abs(H).max()

    return H


def get_image_index(n, m):
    """ Obtains image indexes for an NxM image as column vectors 
        Inputs:
            n - number of columns (x dimension)
            m - number of rows (y dimensions)
    """
    x_index, y_index = np.meshgrid(np.arange(-(n//2), (n//2)), np.arange(-(m//2), (m//2)))
    x_index = x_index.reshape((-1, 1))
    y_index = y_index.reshape((-1, 1))

    return x_index, y_index


def motion_blur(image, a, b):
    """ Obtains motion blurred image 
        Inputs:
            image - original image
            a - horizontal motion factor (0-0.2)
            b - vertical motion factor (0-0.2)
        Outputs:
            blurred_image - blurred image
            H - Frequency Domain Blurring Transform
    """
    if (a == 0) and (b == 0):
        return image 
    
    # compute FFT of image
    F = np.fft.fftshift(np.fft.fft2(image))

    # compute motion blurr function H in Frequency Domain
    n,m = image.shape[:2]
    u_index, v_index = get_image_index(n, m)
    H = motion_transform(u_index, v_index, a, b, T=1).reshape((n, m))
    
    # perform motion blurring in Frequency Domain
    G = F*H

    # get blurred image 
    return np.abs(np.fft.ifft2(G)), H