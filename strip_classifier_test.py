# File: strip_classifier_test.py
# Author: @MichaelHannalla
# Project: Trurapid COVID-19 Strips Detection Server with Python
# Description: Python file for testing the strip classifier based on PyTorch on a single input image

import cv2
import torch
import torch.nn.functional as F

from torch import nn, optim
from utils import classify_crop, get_image_tensor, get_label_from_onehot, strip_dataloader, input_layer_dim

def main():

    # Load the pytorch model
    model = torch.load('models/strip_classifier_mini.pth')
    model.eval()
    
    # Load image
    img = cv2.imread('data/crops/train/n9.jpg')
    label = classify_crop(img, model)
    print(label)    

if __name__ == "__main__":
    main()