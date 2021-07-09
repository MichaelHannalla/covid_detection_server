# Import Computer Vision package - cv2
import cv2

# Import Numerical Python package - numpy as np
import numpy as np
import random as rng

def nothing(x):
    pass #Null Operation

cv2.namedWindow('Trackbars')
# create trackbars for color change
#This is only used for finding the value
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

#After knowing the values of H S & V we can initialize them here
l_h=73
l_s=31
l_v=0
u_h=123
u_s=177
u_v=90

while True:
    frame = cv2.imread("data/train/p11.jpg")
    frame = cv2.resize(frame, (frame.shape[1]//6, frame.shape[0]//6))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower_ = np.array([l_h, l_s, l_v])
    upper_ = np.array([u_h, u_s, u_v])
    kernel_erosion = np.ones((15, 15), np.uint8) #Window size must be odd
    kernel_dilation=np.ones((5,5), np.uint8)
    mask = cv2.inRange(hsv, lower_, upper_)
    erosion = cv2.erode(mask, kernel_erosion)
    dilation = cv2.dilate(erosion, kernel_dilation)
    blurred=cv2.blur(dilation, (1,1))
    
    res_m = cv2.bitwise_and(frame,frame, mask= mask)
    res_b = cv2.bitwise_and(frame,frame, mask= blurred)

    ## [forContour]
    #frames_list.append(frame)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',frame)
    cv2.imshow('result mask',res_m)
    cv2.imshow('result blurred',res_b)
    #cv2.imshow('Denoise',blurred)
    
    # Check if the user has pressed Esc key
    c = cv2.waitKey(1)
    if c == 27:
        break

# Close the capturing device
#for x in frames_list:
    #video_writer.write(x)
cv2.destroyAllWindows()
