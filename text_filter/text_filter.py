import numpy as np
import cv2


# read image as grayscale
image = cv2.imread(r"capture.png", -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blurr image
blurred = cv2.GaussianBlur(image, (15,15), 0)

# threshold image
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

# apply Morphological operations
kernel_3 = np.ones((3,3), dtype=np.uint8)
kernel_5 = np.ones((5,5), dtype=np.uint8)

# perform Morphological Operations
dilation = cv2.dilate(thresh, kernel_5, iterations=1) 
blackhat = cv2.morphologyEx(dilation, cv2.MORPH_BLACKHAT, kernel_3) 
weighted = cv2.addWeighted(dilation, 1.0, blackhat, -1.0, 0) 
erosion = cv2.erode(weighted, kernel_5, iterations=1) 


# Use the simple blob detector to remove small unwanted artifacts
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 250

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(erosion)

# (OPTIONAL) Draw detected blobs as red circles.
im_with_keypoints = cv2.drawKeypoints(erosion, keypoints, np.array([]), (0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# instead use the key points to erase unwanted blobs
filtered = erosion.copy()
for kp in keypoints:
    cv2.circle(filtered, 
               center=np.round(kp.pt).astype(int), 
               radius=np.ceil(kp.size).astype(int), 
               color=(255), 
               thickness=-1)

# display
cv2.imshow("image", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
