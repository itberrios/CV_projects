"""
Image Noise Model
Contains functions to generate iid and correlated image noise

Assumes that a crf is available here and it loads the file once to create the crf and icrf
The crf file contains the average Irradiance and Brightness curves for 190 cameras.
"""

import os
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# load camera response function file
df = pd.read_csv("crf.csv")
I = df["Scene Irradiance"].to_numpy()
B = df["Image Brightness"].to_numpy()


# Get CRF and Inverse CRF
CRF = interp1d(I, B, kind='cubic')
ICRF = interp1d(B, I, kind='cubic')



def inverse_bayer(image_rgb):
    """ Obtains inverse Bayer filtered image
        Inputs:
            image_rgb - (array) rgb image to be filtered
        Outputs:
            bayer_rgb - RGB channel inverse Bayer Bayer image
        """
    (R, G, B) = cv2.split(image_rgb)

    (height, width) = image_rgb.shape[:2]
    bayer = np.empty((height, width), np.uint8)

    # strided slicing for this pattern:
    #   G R
    #   B G
    bayer[0::2, 0::2] = G[0::2, 0::2] # top left
    bayer[0::2, 1::2] = R[0::2, 1::2] # top right
    bayer[1::2, 0::2] = B[1::2, 0::2] # bottom left
    bayer[1::2, 1::2] = G[1::2, 1::2] # bottom right
    #bayer = cv2.resize(bayer, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

    bayer_rgb = cv2.cvtColor(bayer, cv2.COLOR_GRAY2RGB)  # Convert from Grayscale to RGB (r=g=b for each pixel).
    bayer_rgb[0::2, 0::2, 0::2] = 0  # Green pixels - set the blue and the red planes to zero (and keep the green)
    bayer_rgb[0::2, 1::2, 1:] = 0   # Red pixels - set the blue and the green planes to zero (and keep the red)
    bayer_rgb[1::2, 0::2, 0:2] = 0    # Blue pixels - set the red and the green planes to zero (and keep the blue)
    bayer_rgb[1::2, 1::2, 0::2] = 0  # Green pixels - set the blue and the red planes to zero (and keep the green)

    return bayer_rgb


def add_iid_noise(image_rgb, s_sigma=0.02, c_sigma=0.02, c=10):
    """ Obtains iid noise for the respective image input.
        Inputs:
            image - (array) RGB image to add noise to (can also be single channel)
            s_sigma - Std Dev of noise process dependent on irradiance
            c_sigma - Std Dev of noise process independent of irradiance
            c - scale factor for ns to prevent underflow from multiplying values < 0
        Outputs:
            noise - (array) iid noise normalized to 0-255 (uint8)
        """
    # normalize image
    image = cv2.normalize(image_rgb, None, alpha=0, beta=1, 
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # get scene irradiance
    irradiance = ICRF(image)
    
    # Generate Random Noise 
    ns = np.random.normal(0, s_sigma, size=irradiance.shape)*np.sqrt(irradiance)*c
    nc = np.random.normal(0, c_sigma, size=irradiance.shape)

    # get Independent Identically Distributed (iid) noise on the image
    iid_noise = CRF(np.clip(irradiance + ns + nc, 0, 1))
    return cv2.normalize(iid_noise, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def get_camera_noise(image_rgb, s_sigma=0.02, c_sigma=0.02, c=10):
    """ Obtains Camera noise for the respective image input. Camera
        Noise referes to noise added according to a camera model that 
        considers noise to come from 2 separate processes. One that depends
        on scene irradiance (ns) and another the is independent (nc), since
        both noise sources are added before Demosaicing, demosaicing is simulated
        and introduced spatial correlations in the noise.
        Inputs:
            image_rgb - (array) RGB image to add noise to (can also be single channel)
            s_sigma - Std Dev of noise process dependent on irradiance
            c_sigma - Std Dev of noise process independent of irradiance
            c - scale factor for ns to prevent underflow from multiplying values < 0
        Outputs:
            noise - (array) correlated camera noise 
        """
    iid_noise_image = add_iid_noise(image_rgb, s_sigma, c_sigma, c)

    # perform inverse Bayer Filtering if image is RGB
    if len(image_rgb.shape) == 3:
        bayer = cv2.cvtColor(inverse_bayer(image_rgb), cv2.COLOR_RGB2GRAY)
        bayer_noise = cv2.cvtColor(inverse_bayer(iid_noise_image), cv2.COLOR_RGB2GRAY)

         # Demosaic inverse Bayer Filtered images
        demosc = cv2.demosaicing(bayer, dst=None, code=cv2.COLOR_BAYER_GRBG2RGB, dstCn=3)
        demosc_noise = cv2.demosaicing(bayer_noise, dst=None, code=cv2.COLOR_BAYER_GRBG2RGB, dstCn=3)

    else:
         # Demosaic Gray Scale images to correlate the noise
        demosc = cv2.cvtColor(
            cv2.demosaicing(image_rgb, dst=None, code=cv2.COLOR_BAYER_GRBG2RGB), 
            cv2.COLOR_RGB2GRAY)
        demosc_noise = cv2.cvtColor(
            cv2.demosaicing(iid_noise_image, dst=None, code=cv2.COLOR_BAYER_GRBG2RGB), 
            cv2.COLOR_RGB2GRAY)

    # subtract the image from the noisy image, leaving only thr noise
    return demosc_noise - demosc