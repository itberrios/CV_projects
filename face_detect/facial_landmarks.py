"""
Example Script of Facial Extraction pipeline. To be later used for the actual facial extraction

References:
    - https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    - https://github.com/davisking/dlib-models/blob/master
    - http://dlib.net/imaging.html#shape_predictor
"""
# import libraries
import os
import time
import numpy as np
import cv2
import dlib

# local imports
from yunet import YuNet
from utils import *

# ============================================================================
## constants
MODEL_PATH = r"./models"

_FACE_MODEL = "face_detection_yunet_2023mar_int8.onnx"
_FACE_LANDM = "shape_predictor_68_face_landmarks_GTX.dat"

FACE_MODEL = os.path.join(MODEL_PATH, _FACE_MODEL)
FACE_LANDM = os.path.join(MODEL_PATH, _FACE_LANDM)

# DEVICE_ID = 0 
DEVICE_ID = "short_video.avi"

COLOR = (0, 255, 0)

# drawing Options
DRAW_FACE_BOXES = False
DRAW_FACE_REGIONS = False
DRAW_FACE_LANDMARKS = True
DRAW_FACE_CONTOURS = False
MASK_FACE = False
FULL_SIGNAL_MASK = False # requires MASK_FACE = True
REGION_SIGNAL_MASK = False

# ============================================================================
## initialize models

face_model = YuNet(modelPath=FACE_MODEL,
                   inputSize=[320, 320],
                   confThreshold=0.5,
                   nmsThreshold=0.3,
                   topK=100,
                   backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                   targetId=cv2.dnn.DNN_TARGET_CPU)

# dlib ERT model
landmark_predictor = dlib.shape_predictor(FACE_LANDM)


# ============================================================================
# main loop

if __name__ == '__main__':
    cap = cv2.VideoCapture(DEVICE_ID)

    # # try to increase base frame rate
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


    # set input size
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    face_model.setInputSize((w, h))

    # let camera warmup
    time.sleep(2) 

    cnt = 0
    face_detections = None

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    fps = 0
    tic = cv2.getTickCount() / cv2.getTickFrequency()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame Exiting ...")
            break

        # get grayscale frame and 3 channel mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # compute frames per second
        if cnt % 5 == 0:
            toc = cv2.getTickCount() / cv2.getTickFrequency()
            fps = cnt / (toc - tic)
            # reset
            cnt = 0
            tic = cv2.getTickCount() / cv2.getTickFrequency()


        # ====================================================================
            
        # get face predictions
        face_detections = face_model.infer(frame)

        # process frame if any faces are detected
        if len(face_detections) > 0:
            
            # get Facial Landmarks for each face
            for pred in face_detections:
                face_box = pred[:4] # box in xywh format
                landmarks = get_facial_landmarks(landmark_predictor, 
                                                 gray, face_box, extend=True)
                
                # NOTE: This approach is not reliable
                # # get cheek and forhead patches
                # forehead = forehead_from_box(face_box)
                # left_cheek, right_cheek = cheeks_from_box(face_box)

                # draw stuff
                if DRAW_FACE_BOXES:
                    draw_yunet_detections(frame, face_detections, COLOR)

                if DRAW_FACE_REGIONS:
                    # for region in [forehead, left_cheek, right_cheek]:
                    #     x1, y1, x2, y2 = region
                    #     frame = cv2.rectangle(frame, (x1, y1), (x2, y2), 
                    #                           (255, 255, 0), 2)
                    connections_list = [LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD]
                    frame = draw_lines(frame, landmarks, 
                                       connections_list, color=(0, 255, 0))


                if DRAW_FACE_CONTOURS:
                    connections_list = [FACE_OVAL, LEFT_EYE, RIGHT_EYE, MOUTH]
                    frame = draw_lines(frame, landmarks, 
                                       connections_list, color=(0, 255, 0))
                    
                if DRAW_FACE_LANDMARKS:
                    for (x, y) in landmarks:
                        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                if MASK_FACE:
                    mask = np.zeros_like(frame)
                    poly = get_polygon(landmarks, FACE_OVAL)
                    mask = cv2.fillConvexPoly(mask, poly, color=(255,255,255))
                    
                    # remove eyes and mouth
                    if FULL_SIGNAL_MASK:
                        for connections in [LEFT_EYE, RIGHT_EYE, MOUTH]:
                            poly = get_polygon(landmarks, connections)
                            mask = cv2.fillConvexPoly(mask, poly, color=(0,0,0))

                    # mask the frame
                    frame &= mask


                # get only forehead and cheek regions
                if REGION_SIGNAL_MASK:
                    mask = np.zeros_like(frame)
                    for connections in [LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD]:
                        poly = get_polygon(landmarks, connections)
                        mask = cv2.fillConvexPoly(mask, poly, color=(255,255,255))

                    # mask the frame
                    frame &= mask

        # ====================================================================
        # extract signal from frame 
        # blue = frame[:, :, 0]
        # green = frame[:, :, 1]
        # red = frame[:, :, 2]

        # blue = blue.sum()/np.count_nonzero(blue)
        # green = green.sum()/np.count_nonzero(green)
        # red = red.sum()/np.count_nonzero(red)


        # # draw signal levels on frame
        # cv2.putText(frame, "Red: {:.2f}".format(red), (50, 75),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 3)
        # cv2.putText(frame, "Green: {:.2f}".format(green), (50, 100),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 3)
        # cv2.putText(frame, "Blue: {:.2f}".format(blue), (50, 125),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,128,0), 3)

        # draw fps on frame
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (50, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR, 4)

        # display frame
        cv2.imshow('frame', frame)

        # press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        cnt += 1

        # TEMP: slow video down
        time.sleep(0.01)

    # when finished, release the capture
    cap.release()
    cv2.destroyAllWindows()
    del cap
