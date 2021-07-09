import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    
    img = cv2.imread("data/train/p1.jpg", 0)
    img2 = img.copy()
    template = cv2.imread("data/template/template_strip.jpg", 0)
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    img = img2.copy()
    method = eval('cv2.TM_CCOEFF')
    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 2)

    cropped = img[top_left[0]:top_left[0] + h, top_left[1]:top_left[1]+w]
    plt.imshow(cropped)
    plt.show()
 
    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.imshow(img)
    # plt.show()

if __name__ == "__main__":
    main()