""" Implementation of Color Magnification from Eulerian Video Processing 

http://people.csail.mit.edu/mrub/papers/vidmag.pdf


Isaac Berrios
1/26/2024

TODO:
    Add functionality to use camera
    Resize all images to be divisible by 2
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


## Get video frames
def get_frames(video_path, scale_factor):
    """ Obtains Video Frames in YIQ format 
        Inputs:
            video_path - path to video
            scale_factor - scale factor for returned video frames
        Outputs:
            frames - list of frames in YIQ format
            (og_w, og_h) - original width and height of frames
            fs - detected video sample rate
    """
    frames = [] # frames for processing
    cap = cv2.VideoCapture(video_path)

    # video sampling rate
    fs = cap.get(cv2.CAP_PROP_FPS)

    idx = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break

        if idx == 0:
            og_h, og_w, _ = frame.shape
            w = int(og_w*scale_factor)
            h = int(og_h*scale_factor)

        # convert normalized uint8 BGR to the desired color space
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = bgr2yiq(np.float32(frame/255))

        # append resized frame
        frames.append(cv2.resize(frame, (w, h)))

        idx += 1
        
        
    cap.release()
    cv2.destroyAllWindows()
    del cap

    return frames, (og_w, og_h), fs

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


## ==========================================================================================
## construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# Basic Args
ap.add_argument("-v", "--video_path", type=str, required=True,
    help="path to input video")

ap.add_argument("-a", "--alpha", type=float, default=25.0, required=True,
    help="ColorMagnification Factor")

ap.add_argument("-lo", "--freq_lo", type=float, required=True,
    help="Low Frequency cutoff for Temporal Filter")

ap.add_argument("-hi", "--freq_hi", type=float, required=True,
    help="High Frequency cutoff for Temporal Filter")

ap.add_argument("-fs", "--sample_frequency", type=float, default=-1.0,
    help="Video sample frequency, defaults to sample frequency from input "  \
            "video if input is less than or equal to zero. Video is " \
            "reconstructed with detected sample frequency")

ap.add_argument("-l", "--level", type=int, default=4,
    help="Gaussian Pyramid Level, we only use a single level for processing")

ap.add_argument("-c", "--scale_factor", type=float, default=1.0,
    help="Scales down image to rpeserve memory")

ap.add_argument("-d", "--save_directory", type=str, default="",
    help="Save directory for output video or GIF, if False outputs \
        are placed in the same location as the input video")

ap.add_argument("-gif", "--save_gif", type=bool, default=False,
    help="Determines whether to save GIF of results")

if __name__ == '__main__':

    ## Default use commandline args 
    ## --> Comment this out to manually input args in script
    # args = vars(ap.parse_args())

    # Optional: Pass arguments directly in script
    # --> Comment this out to receive args from commandline
    args = vars(ap.parse_args(
        ["--video_path",       r"C:\Users\itber\Documents\learning\self_tutorials\phase_based\videos\face2.mp4", # "videos/eye.avi", # "videos/crane_crop.avi", 
         "--alpha",            "100", 
         "--freq_lo",          "0.83",
         "--freq_hi",          "1.0", 
         "--sample_frequency", "-1.0", 
         "--level",            "6",
         "--scale_factor",     "0.4", # "1.0", # this may help troubleshoot image shape issues
         "--save_directory",   r"C:\Users\itber\Documents\learning\self_tutorials\CV_projects\color_mag",
         "--save_gif",         "False" # "True"
         ]))

    ## Parse Args   
    video_path       = args["video_path"]
    alpha            = args["alpha"]
    freq_lo          = args["freq_lo"]
    freq_hi          = args["freq_hi"]
    sample_frequency = args["sample_frequency"]
    level            = args["level"]
    scale_factor     = args["scale_factor"]
    save_directory   = args["save_directory"]
    save_gif         = args["save_gif"]

    ## ======================================================================================
    ## start the clock once the args are received
    tic = cv2.getTickCount()

    ## ======================================================================================
    ## Process input filepaths
    if not os.path.exists(video_path):
        print(f"\nInput video path: {video_path} not found! exiting \n")
        sys.exit()
        
    if not save_directory:
        save_directory = os.path.dirname(video_path)
    elif not os.path.exists(save_directory):
        save_directory = os.path.dirname(video_path)
        print(f"\nSave Directory not found, " \
               "using default input video directory instead \n")
    
    video_name = re.search("\w*(?=\.\w*)", video_path).group()
    video_output = f"{video_name}_{level}_{int(alpha)}x.mp4"
    video_save_path = os.path.join(save_directory, video_output)

    print(f"\nProcessing {video_name} " \
          f"and saving results to {video_save_path} \n")
    


    ## Get video frames
    frames, (og_w, og_h), fs = get_frames(video_path, scale_factor)
    num_frames = len(frames)

    ## Get sample rate
    print(f"Detected Video Sampling rate: {fs}")

    if sample_frequency > 0:
        print(f"Overriding to: {sample_frequency}")
        fs = sample_frequency
        video_fs = fs
    else:
        video_fs = fs

    ## Get Temporal Filter
    bandpass = signal.firwin(numtaps=num_frames,
                             cutoff=(freq_lo, freq_hi),
                             fs=video_fs,
                             pass_zero=False)
    
    transfer_function = np.fft.fft(np.fft.ifftshift(bandpass))

    ## Get Gaussian Pyramid Stack
    rows, cols, colors = frames[0].shape
    scale = 2**level
    pyramid_stack = np.zeros((num_frames, colors, rows//scale, cols//scale))

    for i, frame in enumerate(frames):
        pyramid = gaussian_pyramid(frame, level)
        pyramid_stack[i, :, :, :] = pyramid

    ## Apply Temporal Filtering
    pyr_stack_fft = np.fft.fft(pyramid_stack, axis=0).astype(np.complex64)
    _filtered_pyramid = pyr_stack_fft * transfer_function[:, None, None, None].astype(np.complex64)
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

    # SUPER TEMP
    og_reds = []
    og_blues = []
    og_greens = []

    reds = []
    blues = []
    greens = []
    for i in range(num_frames):
        # convert YIQ to RGB
        frame = inv_colorspace(frames[i])
        og_reds.append(frame[0, :, :].sum())
        og_blues.append(frame[1, :, :].sum())
        og_greens.append(frame[2, :, :].sum())

        reds.append(magnified[i][0, :, :].sum())
        blues.append(magnified[i][1, :, :].sum())
        greens.append(magnified[i][2, :, :].sum())

    times = np.arange(0, num_frames)/fs
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    ax[0].plot(times, og_reds, color='red')
    ax[0].plot(times, og_blues, color='blue')
    ax[0].plot(times, og_greens, color='green')
    ax[0].set_title("Original", size=18)
    ax[0].set_xlabel("Time", size=16)
    ax[0].set_ylabel("Intensity", size=16)

    ax[1].plot(times, reds, color='red')
    ax[1].plot(times, blues, color='blue')
    ax[1].plot(times, greens, color='green')
    ax[1].set_title("Filtered", size=18)
    ax[1].set_xlabel("Time", size=16);
    plt.show();
    
    ## ======================================================================================
    ## Process results
        
    w, h, _ = mag.shape

    ## get stacked side-by-side comparison frames
    og_h = int(h/scale_factor)
    og_w = int(w/scale_factor)
    middle = np.zeros((og_h, 3, 3)).astype(np.uint8)

    stacked_frames = []

    for vid_idx in range(num_frames):

        og_frame = cv2.normalize(yiq2rgb(frames[vid_idx]), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)

        # get resized frames
        og_frame = cv2.resize(
            cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR), (og_w, og_h))
        processed = cv2.resize(
            cv2.cvtColor(magnified[vid_idx], cv2.COLOR_RGB2BGR), (og_w, og_h))

        # stack frames
        stacked = np.hstack((og_frame, 
                             middle, 
                             processed))

        stacked_frames.append(stacked)


    ## ======================================================================================
    ## make video
    # get width and height for stacked video frames
    sh, sw, _ = stacked_frames[-1].shape

    # save to mp4
    out = cv2.VideoWriter(video_save_path,
                          cv2.VideoWriter_fourcc(*'MP4V'), 
                          int(np.round(video_fs)), 
                          (sw, sh))
    
    for frame in stacked_frames:
        out.write(frame)

    out.release()
    del out

    print(f"Result video saved to: {video_save_path} \n")

    ## ======================================================================================
    ## make GIF if desired
    if save_gif:
        
        # replace video extension with ".gif"
        gif_save_path = re.sub("\.\w+(?<=\w)", ".gif", video_save_path)

        print(f"Saving GIF to: {gif_save_path} \n")

        # size back down for GIF
        sh = int(sh*scale_factor)
        sw = int(sw*scale_factor)

        # accumulate PIL image objects
        pil_images = []
        for img in stacked_frames:
            img = cv2.cvtColor(cv2.resize(img, (sw, sh)), cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img))

        # create GIF
        pil_images[0].save(gif_save_path, 
                           format="GIF", 
                           append_images=pil_images, 
                           save_all=True, 
                           duration=50, # duration that each frame is displayed
                           loop=0)

    ## ======================================================================================
    ## end of processing
        
    # get time elapsed in Hours : Minutes : Seconds
    toc = cv2.getTickCount()
    time_elapsed = (toc - tic) / cv2.getTickFrequency()
    time_elapsed = str(datetime.timedelta(seconds=time_elapsed))

    print("Motion Magnification processing complete! \n")
    print(f"Time Elapsed (HH:MM:SS): {time_elapsed} \n")

