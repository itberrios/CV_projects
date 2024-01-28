"""
This scipt contains functions to perform Color Magnification 
on a list of RGB video frames

Isaac Berrios
January 2024

"""

import os
import sys
from glob import glob
import re
import datetime
import argparse
from PIL import Image
import numpy as np 
import cv2
import scipy.signal as signal
import matplotlib.pyplot as plt 


## ======================================================================================
## Helper functions

## Color spaces
def rgb2yiq(rgb):
    """ Converts an RGB image to YIQ using FCC NTSC format.
        This is a numpy version of the colorsys implementation
        https://github.com/python/cpython/blob/main/Lib/colorsys.py
        Inputs:
            rgb - (N,M,3) rgb image
        Outputs
            yiq - (N,M,3) YIQ image
        """
    # compute Luma Channel
    y = rgb @ np.array([[0.30], [0.59], [0.11]])

    # subtract y channel from red and blue channels
    rby = rgb[:, :, (0,2)] - y

    i = np.sum(rby * np.array([[[0.74, -0.27]]]), axis=-1)
    q = np.sum(rby * np.array([[[0.48, 0.41]]]), axis=-1)

    yiq = np.dstack((y.squeeze(), i, q))
    
    return yiq


def bgr2yiq(bgr):
    """ Coverts a BGR image to float32 YIQ """
    # get normalized YIQ frame
    rgb = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    yiq = rgb2yiq(rgb)

    return yiq


def yiq2rgb(yiq):
    """ Converts a YIQ image to RGB.
        Inputs:
            yiq - (N,M,3) YIQ image
        Outputs:
            rgb - (N,M,3) rgb image
        """
    r = yiq @ np.array([1.0, 0.9468822170900693, 0.6235565819861433])
    g = yiq @ np.array([1.0, -0.27478764629897834, -0.6356910791873801])
    b = yiq @ np.array([1.0, -1.1085450346420322, 1.7090069284064666])
    rgb = np.clip(np.dstack((r, g, b)), 0, 1)
    return rgb


inv_colorspace = lambda x: cv2.normalize(
    yiq2rgb(x), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)

## Gaussian Pyramid
def gaussian_pyramid(image, level):
    """ Obtains single band of a Gaussian Pyramid Decomposition
        Inputs: 
            image - single channel input image
            num_levels - number of pyramid levels
        Outputs:
            pyramid - Pyramid decomposition tensor
        """ 
    rows, cols, colors = image.shape
    scale = 2**level # downscale factor
    pyramid = np.zeros((colors, rows//scale, cols//scale))

    for i in range(0, level):

        image = cv2.pyrDown(image, dstsize=(cols//2, rows//2))
        rows, cols, _ = image.shape

        if i == (level - 1):
            for c in range(colors):
                pyramid[c, :, :] = image[:, :, c]

    return pyramid


## Color mag function
def mag_colors(rgb_frames, fs, freq_lo, freq_hi, level, alpha):
    """ Function to obtain Amplified Colors in a given list of RGB frames 
        Inputs:
            rgb_frames - list of RGB video frames
            fs - sample frequency
            freq_lo - lower frequency bound
            freq_hi - upper frequency bound
            level - level of Gaussian Pyramid
            alpha - magnification factor
        Outputs:
            magnified_frames - COlor magnified RGB video frames
    """
    rows, cols, colors = rgb_frames[0].shape
    num_frames = len(rgb_frames)

    # convert frames to YIQ colorspace
    frames = [rgb2yiq(frame) for frame in rgb_frames]

    ## Get Temporal Filter
    bandpass = signal.firwin(numtaps=num_frames,
                             cutoff=(freq_lo, freq_hi),
                             fs=fs,
                             pass_zero=False)
    
    transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))
    transfer_function = transfer_function[:, None, None, None].astype(np.complex64)

    ## Get Gaussian Pyramid Stack
    scale = 2**level
    pyramid_stack = np.zeros((num_frames, colors, rows//scale, cols//scale))
    for i, frame in enumerate(frames):
        pyramid = gaussian_pyramid(frame, level)
        pyramid_stack[i, :, :, :] = pyramid

    ## Apply Temporal Filtering
    pyr_stack_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)
    _filtered_pyramid = pyr_stack_fft * transfer_function
    filtered_pyramid = np.fft.ifft(_filtered_pyramid, axis=0).real

    ## Apply magnification to video
    magnified_pyramid = filtered_pyramid * alpha

    ## Collapse Pyramid and reconstruct video
    magnified = []

    for i in range(num_frames):
        y_chan = frames[i][:, :, 0] 
        i_chan = frames[i][:, :, 1] 
        q_chan = frames[i][:, :, 2] 
        
        fy_chan = cv2.resize(magnified_pyramid[i, 0, :, :], (cols, rows))
        fi_chan = cv2.resize(magnified_pyramid[i, 1, :, :], (cols, rows))
        fq_chan = cv2.resize(magnified_pyramid[i, 2, :, :], (cols, rows))

        # apply magnification
        mag = np.dstack((
            y_chan + fy_chan,
            i_chan + fi_chan,
            q_chan + fq_chan,
        ))

        # convert to RGB and normalize
        mag = inv_colorspace(mag)

        # store magnified frames
        magnified.append(mag)

    return magnified
    