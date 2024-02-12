""" Example of Face Detection """

import os
import time
import numpy as np
import cv2

from utils import *
from yunet import YuNet

## ===========================================================================
# settings
YUNET = False # True # use YuNet else use SSD detector
DRAW = True  # draw bounding boxes
COLOR = (255, 255, 0) # color for bounding boxes

## ===========================================================================
# get model paths
MODEL_PATH = r"./models"
_FACE_MODEL_YUNET = "face_detection_yunet_2023mar_int8.onnx"
_FACE_MODEL_SSD = "res10_300x300_ssd_iter_140000.caffemodel"
_FACE_PROTO_SSD = "deploy.prototxt.txt"



# get face model
if YUNET:
    FACE_MODEL = os.path.join(MODEL_PATH, _FACE_MODEL_YUNET)
    face_model = YuNet(modelPath=FACE_MODEL,
                       inputSize=[320, 320],
                       confThreshold=0.5,
                       nmsThreshold=0.3,
                       topK=100, # set number of detections
                       backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                       targetId=cv2.dnn.DNN_TARGET_CPU)
else:
    FACE_MODEL = os.path.join(MODEL_PATH, _FACE_MODEL_SSD)
    FACE_PROTO = os.path.join(MODEL_PATH, _FACE_PROTO_SSD)
    face_model = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)


## ===========================================================================
## main program

if __name__ == '__main__':
    deviceId = 0
    # deviceId = "/home/iberrios/Documents/projects/color_mag/cap.mp4
    # deviceId = "notebooks/short_video.avi"
    cap = cv2.VideoCapture(deviceId)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # set input size for YuNet  
    if YUNET:
        face_model.setInputSize((w, h))

    time.sleep(2) # let camera warmup

    # video sampling rate
    # fs = cap.get(cv2.CAP_PROP_FPS)

    cnt = 0

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    fps = 0
    tic = time.perf_counter() # cv2.getTickCount() / cv2.getTickFrequency()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame Exiting ...")
            break

        # compute frames per second
        if cnt % 10 == 0:
            toc = time.perf_counter() # cv2.getTickCount() / cv2.getTickFrequency()
            fps = cnt / (toc - tic)
            # reset
            cnt = 0
            tic = time.perf_counter() # cv2.getTickCount() / cv2.getTickFrequency()

        # get face detections
        if YUNET:
            face_detections = face_model.infer(frame)
            draw_yunet_detections(frame, face_detections, COLOR)
        else:
            detections = get_face_detections_ssd(face_model, frame, w, h, confidence_thresh=0.5)
            draw_faces_ssd(frame, detections, COLOR)

        # draw fps on frame
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (50, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 4)

        # display frame
        cv2.imshow('frame', frame)

        # press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        cnt += 1

    # when finished, release the capture
    cap.release()
    cv2.destroyAllWindows()
    del cap