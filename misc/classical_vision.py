import numpy as np
import cv2
import utils

class COVIDStripDetector():
    def __init__(self):
        pass

def nothing(x):
    pass #Null Operation

def main():
    
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar("LOW", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("HIGH", "Trackbars", 255, 255, nothing)

    img = cv2.imread("data/train/p11.jpg")
    img = cv2.resize(img, (img.shape[1]//6, img.shape[0]//6))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_color_processed = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    while True:

        low = cv2.getTrackbarPos("LOW", "Trackbars")
        high = cv2.getTrackbarPos("HIGH", "Trackbars")
        edges = cv2.Canny(img_color_processed, low, high, L2gradient=True)


        # (mu, sigma) = cv2.meanStdDev(result)
        # edges = cv2.Canny(result, int(mu - 4*sigma), int(mu + sigma))
        # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, None, 1, 1)
        # blank = np.zeros_like(edges).copy()

        # if linesP is not None:
        #     print(linesP)
        #     for i in range(0, len(linesP)):
        #         l = linesP[i][0]
        #         blank = cv2.line(blank, (l[0], l[1]), (l[2], l[3]), (255), 3, cv2.LINE_AA)
        

        cv2.imshow("img", edges)
        c = cv2.waitKey(0)
        if c == 27:
            break

if __name__ == "__main__":
    main()