# File: strip_classifier_classical.py
# Author: @MichaelHannalla
# Project: Trurapid COVID-19 Strips Detection Server with Python
# Description: Python file for testing the strip classification using classical vision techniques

import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn, optim
from utils import classify_crop, get_image_tensor, get_label_from_onehot, strip_dataloader, input_layer_dim
from utils import null_func, negate_image


from scipy.signal import savgol_filter

def main():
    
    # Load image
    img = cv2.imread('data/doubtful/reader-48.jpg') 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # convert to grayscale
    img = negate_image(img)
    
    intensities_vec = img[:, img.shape[1]//2]              
    intensities_vec = savgol_filter(intensities_vec, 25, 2)     # second order savistky-golay filter with 25 window size
    grad_vec = np.gradient(intensities_vec)
    
    #grad_vec = savgol_filter(grad_vec, 25, 2)                   # second order savistky-golay filter with 25 window size
    grad_vec[np.absolute(grad_vec) < 0.5] = 0
    grad_positive_check = grad_vec[len(grad_vec)//2: int(3 * len(grad_vec)//4)]

    plt.ylim([-10, 10])
    plt.plot(grad_positive_check)
    plt.show()
    
    
if __name__ == "__main__":
    main()