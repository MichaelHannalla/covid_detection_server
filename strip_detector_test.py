# File: strip_detector_test.py
# Author: @MichaelHannalla
# Project: Trurapid COVID-19 Strips Detection Server with Python
# Description: Python file for testing the strip detector based on detecto on a single input image

from detecto.core import Model
from detecto.utils import read_image
from detecto.visualize import plot_prediction_grid
from utils import get_crop
import matplotlib.pyplot as plt
import cv2

def main():

    model = Model.load('models/strip_detector_weights_pass2.pth', ['strip'])
    negative1 = read_image('data/test/strip11.jpeg')
    positive1 = read_image('data/test/strip21.jpeg')
    positive2 = read_image('data/test/strip41.jpeg')

    # images = [image3]
    labels, boxes, scores = model.predict(positive1)
    cropped = get_crop(positive1, boxes[0])
    cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    cv2.imshow('out', cropped)
    cv2.waitKey(0)

    #plot_prediction_grid(model, images, dim=None, figsize=None, score_filter=0.6)
    
if __name__ == "__main__":
    main()
    