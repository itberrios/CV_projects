"""
    Perform Motion Detection with Sparse Optical Flow

"""

import os
from glob import glob
import re
import numpy as np
import cv2
from scipy.stats import kurtosis
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from motion_detection_utils import *


def motion_comp(prev_frame, curr_frame, num_points=500, points_to_use=500, transform_type='affine'):
    """ Obtains new warped frame1 to account for camera (ego) motion
        Inputs:
            prev_frame - first image frame
            curr_frame - second sequential image frame
            num_points - number of feature points to obtain from the images
            points_to_use - number of point to use for motion translation estimation 
            transform_type - type of transform to use: either 'affine' or 'homography'
        Outputs:
            A - estimated motion translation matrix or homography matrix
            prev_points - feature points obtained on previous image
            curr_points - feature points obtaine on current image
        """
    transform_type = transform_type.lower()
    assert(transform_type in ['affine', 'homography'])

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

    # get features for first frame
    corners = cv2.goodFeaturesToTrack(prev_gray, num_points, qualityLevel=0.01, minDistance=10)

    # get matching features in next frame with Sparse Optical Flow Estimation
    matched_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)

    # reformat previous and current corner points
    prev_points = corners[status==1]
    curr_points = matched_corners[status==1]

    # sub sample number of points so we don't overfit
    if points_to_use > prev_points.shape[0]:
        points_to_use = prev_points.shape[0]

    index = np.random.choice(prev_points.shape[0], size=points_to_use, replace=False)
    prev_points_used = prev_points[index]
    curr_points_used = curr_points[index]

    # find transformation matrix from frame 1 to frame 2
    if transform_type == 'affine':
        A, _ = cv2.estimateAffine2D(prev_points_used, curr_points_used, method=cv2.RANSAC)
    elif transform_type == 'homography':
        A, _ = cv2.findHomography(prev_points_used, curr_points_used)

    return A, prev_points, curr_points


def get_motion_detections(frame1, 
                          frame2, 
                          cluster_model, 
                          c=2, 
                          angle_thresh=0.1, 
                          edge_thresh=50, 
                          max_cluster_size=80, 
                          distance_metric='l2', 
                          transform_type='affine'):
    """ Obtains detected motion betweej frame 1 and frame 2 
        Inputs:
            frame1 - previous frame
            frame2 - current frame
            cluster_model - cluster model object
            c - tunable threshold hyperparamer for outlier detection
            angle_thresh - threshold for angular uniformity of the cluster 
                (Determines if the Std Dev of the Cluster flow angles is too large)
            edge_thresh - min pixel distance to edge for cluster removal
                clusters close to the edge usually correspond to false detections
            max_cluster_size - max number of points for a cluster
            distance_metric - used to compute the distance between current and previous keypoints
            transform_type - type of transform to use: either 'affine' or 'homography'
        Outputs:
            clusters - list containing clusters of detected motion keypoints
        """
    transform_type = transform_type.lower()
    assert(transform_type in ['affine', 'homography'])

    # get frame info
    h, w, _ = frame1.shape

    # get affine transformation matrix for motion compensation between frames
    A, prev_points, curr_points = motion_comp(frame1, frame2, num_points=10000, points_to_use=5000, transform_type=transform_type)

    # get transformed key points
    if transform_type == 'affine':
        A = np.vstack((A, np.zeros((3,)))) # get 3x3 matrix to xform points

    compensated_points = np.hstack((prev_points, np.ones((len(prev_points), 1)))) @ A.T
    compensated_points = compensated_points[:, :2]

    # get a distance metric for the current and previous keypoints
    if distance_metric == 'l1':
        x = np.sum(np.abs(curr_points - compensated_points), axis=1) # l1 norm
    else:
        x = np.linalg.norm(curr_points - compensated_points, ord=2, axis=1) # l2 norm
    
    # # compute kurtosis of x to determine outlier hyperparameter c
    # NOTE: We expect a Letpokurtic distribution with a heavy positive tail
    # If we don't have extremely heavy tails reduce outlier threshold parameter
    if kurtosis(x, bias=False) < 1:
        c /= 2 # reduce outlier hyparameter

    # get outlier bound (only care about upper bound since lower values are not likely movers)
    upper_bound = np.mean(x) + c*np.std(x, ddof=1)

    # get motion points
    motion_idx = (x >= upper_bound)
    motion_points = curr_points[motion_idx]

    # add additional motion data for clustering
    motion = compensated_points[motion_idx] - motion_points
    magnitude = np.linalg.norm(motion, ord=2, axis=1)
    angle = np.arctan2(motion[:, 0], motion[:, 1]) # horizontal/vertial

    motion_data = np.hstack((motion_points, np.c_[magnitude], np.c_[angle]))

    # cluster motion data
    cluster_model.fit(motion_data)

    
    # filter clusters with large variation in angular motion
    clusters = []
    far_edge_array = np.array([w - edge_thresh, h - edge_thresh])
    for lbl in np.unique(cluster_model.labels_):
        
        cluster_idx = cluster_model.labels_ == lbl
        
        # get standard deviation of the angle of apparent motion 
        angle_std = angle[cluster_idx].std(ddof=1)
        if angle_std <= angle_thresh:
            cluster = motion_points[cluster_idx]

            # remove clusters that are too close to the edges and ones that are too large
            centroid = cluster.mean(axis=0)
            if (len(cluster) < max_cluster_size) \
                and not (np.any(centroid < edge_thresh) or np.any(centroid > far_edge_array)):
                clusters.append(cluster)

    return clusters



if __name__ == '__main__':

    # get datapath
    fpath = r"C:\Users\itber\Documents\datasets\VisDrone\VisDrone2019-VID-train\sequences\uav0000352_05980_v"

    # get image paths
    image_paths = sorted(glob(f"{fpath}/*.jpg"), key=lambda x:float(re.findall("(\d+)",x)[0]))

    # get cluster model
    cluster_model = DBSCAN(eps=30.0, min_samples=3)

    # get previous frame
    prev_frame = cv2.imread(image_paths[0])
    curr_frame = None

    # iterate through all frames
    frames = []
    for i in range(1, len(image_paths)):
        curr_frame = cv2.imread(image_paths[i])

        # get detected cluster
        clusters = get_motion_detections(prev_frame, 
                                         curr_frame, 
                                         cluster_model, 
                                         c=1,
                                         angle_thresh=0.1, 
                                         max_cluster_size=50,
                                         distance_metric='l2', 
                                         transform_type='affine')

        # save previous frame for next iteration
        prev_frame = curr_frame.copy()

        ## Display Options
        # draw detected clusters
        for j, cluster in enumerate(clusters):
            color = get_color((j+1)*5)
            curr_frame = plot_points(curr_frame, cluster, radius=10, color=color)

        # save image for GIF
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(cv2.resize(curr_frame, (952, 535)))
        plt.axis('off')
        fig.savefig(f"temp/frame_{i}.png")
        plt.close();
        
        frames.append(curr_frame)

    # create video

    # create GIF
    
