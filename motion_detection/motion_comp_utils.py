"""
    Motion compensation utils
"""

import cv2
import numpy as np


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