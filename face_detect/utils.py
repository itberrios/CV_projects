"""
Isaac Berrios - Feb 3, 2024

Util functions for Face2PPG 

"""

import numpy as np
import cv2
import dlib

from face_mesh import *


## ===========================================================================
# SSD Face detector utils
def get_face_detections_ssd(net : cv2.dnn.Net, 
                            image : np.ndarray, 
                            w : int, 
                            h : int, 
                            confidence_thresh : int = 0.5) -> list:
    """ Obtains face detections from SSD model in (x1, y1, x2, y2, conf) 
        format for a given BGR image.
        NOTE: The SSD face model already performs Non-Maximal Supression 
        during inference
        Inputs:
            model - opencv dnn Net object
            image - BGR image
            w - original image width
            h - original image height
            confidence_thresh - confidence threshold
        Outputs:
            detections - Nx5 face detections array in (x1, y1, x2, y2, conf) format
        """
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 
                                 scalefactor=1.0,
                                 size=(300, 300), 
                                 mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    predictions = net.forward()

    detections = []
    for i in range(0, 10): # predictions.shape[2]):
        # extract confidence
        confidence = predictions[0, 0, i, 2]

        # remove low confidence predictions
        if confidence > confidence_thresh:
            # get (x, y) coordinates of bounding box
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            detections.append([x1, y1, x2, y2, confidence])
            
    return detections




def draw_faces_ssd(image : np.ndarray, detections : list, color : tuple = (255, 255, 0)) -> None:
    """ Draws detected face bounding box and confidence on image 
        Inputs:
            image - image to fraw faces on 
            detections - Nx5 detections list in (x1, y1, x2, y2, conf) format
        Outputs:
            None
        """
    for (x1, y1, x2, y2, conf) in detections:
        text = "{:.2f}%".format(conf * 100)
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(image, (x1, y1), (x2, y2),
            color, 2)
        cv2.putText(image, text, (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

## ===========================================================================
# ref: https://github.com/PyImageSearch/imutils/blob/master/imutils/face_utils/helpers.py
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


## ===========================================================================
## drawing functions

def draw_yunet_detections(image : np.ndarray, 
                          predictions : np.ndarray, 
                          color : tuple = (255, 255, 0)) -> None:
    """ Draws YuNet bounding boxes and confidences 
        Inputs:
            image - BGR image
            predictions - Nx15 predictions array
            color - color tuple
    """
    for det in predictions:
        # get boxes in xywh format
        bbox = det[0:4].astype(np.int32)
        conf = det[-1]

        cv2.rectangle(image, 
                      (bbox[0], bbox[1]), 
                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), 
                      color, 
                      2)
        text = "{:.2f}%".format(conf * 100)
        y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
        cv2.putText(image, text, (bbox[0], y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        

def draw_lines(image_in, landmarks, connections_list, color):
    """ Base function to draw lines on image, and also gathers a list
        of numpy arrays that correspond to polygons for each connection
        list.
        Inputs:
            image_in - 3 channel input image
            landmarks - list of facial landmarks
            connections_list - list of connection point indicies that will 
                determine how the lines are drawn
            color - color to draw lines
        Outputs:
            image - copy of image with drawn lines
        """
    assert isinstance(connections_list, (list, tuple, np.ndarray))
    assert image_in.shape[2] == 3
    
    image = np.copy(image_in)

    for connections in connections_list:
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            cv2.line(image, 
                     landmarks[start_idx], 
                     landmarks[end_idx],
                     color,
                     2)
    return image

def get_polygon(landmarks, connections):
    """ Obtains array of polygon vertices 
        Inputs:
            landmarks - list of facial landmarks
            connections - list of polygon vertex indices
        Outputs:
            poly- umpy array of polygon vertices
    """
    poly = []
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        poly += [landmarks[start_idx], landmarks[end_idx]]

    return np.array(poly)

## ===========================================================================
## landmarks
def get_cheek_landmarks(landmarks, cheek_connections):
    """ Obtains interpolated cheek landmarks 
        Inputs:
            landmarks - original list of 68 facial landmarks
            cheek_connections - cheek connections list
        Outputs:
            cheek_landmarks - list of new cheek landmarks
        """
    cheek_landmarks = []
    for connection in cheek_connections:
        start_idx = connection[0]
        end_idx = connection[1]

        interp_landmark = np.int32(
            landmarks[start_idx] 
            - (landmarks[start_idx] - landmarks[end_idx])/2)

        cheek_landmarks.append(interp_landmark)

    return cheek_landmarks


def get_forehead_landmarks(landmarks, forehead_base):
    """ Obtains interpolated forehead landmarks. Forehad landmarks
        are a 'nose length' above the eyebrows with an offset
        distance for an aestetic cut of the face.
        Inputs:
            landmarks - original list of 68 facial landmarks
            forehead_base- forehead connections list
                (base landmark index, scale factor)
        Outputs:
        """
    nasal_root = landmarks[27]
    nose_length = np.array([0, (landmarks[33] - nasal_root)[1]])

    forehead_landmarks = []
    for idx, scale_factor in forehead_base:
        if idx == 27:
            interp_landmark = landmarks[idx] - nose_length
        else:
            eyebrow_offset = (nasal_root - landmarks[idx])[1]
            interp_landmark = landmarks[idx] - nose_length*scale_factor
            interp_landmark[1] += eyebrow_offset
            
        forehead_landmarks.append(np.int32(interp_landmark))

    return forehead_landmarks
        

def get_facial_landmarks(landmark_predictor, gray, face_pred, extend=True):
    """ Obtains facial landmarks 
        Inputs:
            landmark_predictor - dlib facial landmark predictor
            gray - grayscale image
            face_pred - predicted face bounding box in xywh format
            extend - if True add 17 additional landmarks, else use 68
        Outputs:
            landmarks - Nx2 list of facial landmark points in (x,y) format
        """
    # get 68 facial landmarks
    rect = dlib.rectangle(face_pred[0], 
                          face_pred[1], 
                          face_pred[0] + face_pred[2], 
                          face_pred[1] + face_pred[3])
    shape = landmark_predictor(gray, rect)
    landmarks = list(shape_to_np(shape))

    if extend:
        # get extended landmarks 68-85
        cheek_landamrks = get_cheek_landmarks(landmarks, CHEEK_INTERP_POINTS)
        forehead_landmarks = get_forehead_landmarks(landmarks, FOREHEAD_BASE)

        landmarks += cheek_landamrks
        landmarks += forehead_landmarks

    return landmarks


def forehead_from_box(face_box):
    """ Obtains rough guesstimate of forehead region based on face box
        Inputs:
            face_box - face bounding box in xywh format
        Outputs:
            forehead region in xyxy format
        """
    x, y, w, h = face_box

    x1 = np.int32(x + 0.15*w)
    x2 = np.int32(x + 0.85*w)

    y1 = np.int32(y - 0.0*h)
    y2 = np.int32(y + 0.2*h)

    return (x1, y1, x2, y2)


def cheeks_from_box(face_box):
    """ Obtains rough guesstimate of left and right cheek regiond based on 
        face box
        Inputs:
            face_box - face bounding box in xywh format
        Outputs:
            left and right cheek regions in xyxy format
        """
    x, y, w, h = face_box

    # left
    x1l = np.int32(x + 0.05*w) 
    x2l = np.int32(x + 0.4*w)

    y1l = np.int32(y + 0.4*h)
    y2l = np.int32(y + 0.7*h)

    # right
    x1r = np.int32(x + 0.6*w)
    x2r = np.int32(x + 0.95*w) 

    y1r = np.int32(y + 0.4*h)
    y2r = np.int32(y + 0.7*h)

    left_cheek = (x1l, y1l, x2l, y2l)
    right_cheek = (x1r, y1r, x2r, y2r)

    return left_cheek, right_cheek