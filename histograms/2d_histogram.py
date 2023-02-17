import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

image = cv2.imread("fish.png")[:, :, :3]

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# compute RGB histograms
color = ('b','g','r')
plt.figure()
plt.title("RGB histograms")
for i,col in enumerate(color):
    histr = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


# compute 2D histograms in RGB space
# Blue Green histogram
hist = cv2.calcHist([image], [0,1], None, [256, 256], [0, 256, 0, 256])

# blue green map
bg_map = np.zeros((256, 256, 3))
b, g = np.indices(bg_map.shape[:2])
bg_map[:, :, 0] = 0
bg_map[:, :, 1] = g
bg_map[:, :, 2] = b
hist = np.clip(hist*0.05, 0, 1)
vis = bg_map*hist[:,:,np.newaxis] / 255.0

plt.imshow(vis)
plt.show()


# compute 2D histograms in HSV space
hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
plt.imshow(hist, interpolation='nearest') # , norm=LogNorm(vmin=1, vmax=100))
plt.imshow(hist)
plt.show()

# ====================================================
hsv_map = np.zeros((180, 256, 3), np.uint8)
h, s = np.indices(hsv_map.shape[:2])
hsv_map[:,:,0] = h
hsv_map[:,:,1] = s
hsv_map[:,:,2] = 255
hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)
cv2.imshow('hsv_map', hsv_map)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# dark = hsv[...,2] < 32
# hsv[dark] = 0
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

hist_scale = 10
hist = np.clip(hist*0.005*hist_scale, 0, 1)
vis = hsv_map*hist[:,:,np.newaxis] / 255.0

plt.imshow(vis)
plt.title("HSV Histogram")
plt.xlabel("Saturation")
plt.ylabel("Hue")
plt.show();
