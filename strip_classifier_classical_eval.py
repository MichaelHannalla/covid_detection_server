# File: strip_classifier_positive_eval.py
# Author: @MichaelHannalla
# Project: Trurapid COVID-19 Strips Detection Server with Python
# Description: Python file for testing the strip classifier against a set of positive samples to test how it can accurately get the lighter bands

import os
import cv2
import torch
from utils import get_crop, classical_classify_crop
from detecto.core import Model
from detecto.utils import read_image

def main():

    # Load the pytorch model
    detector = Model.load('models/covid_strip_weights_single_class.pth', ['strip'])

    positives_path = "data/positive_images"
    positives = os.listdir(positives_path)

    err_count = 0

    for idx, positive_img_path in enumerate(positives):
        if positive_img_path.endswith('.jpeg') or positive_img_path.endswith('.jpg'):
            
            positive_img_full_path = positives_path + "/" + positive_img_path
            positive_img = read_image(positive_img_full_path)
            labels, boxes, scores = detector.predict(positive_img)
            strip_crop = get_crop(positive_img, boxes[0])
            label = classical_classify_crop(strip_crop)
            err_count += label

    print("Total error in positive samples: {}%".format(err_count/len(positives)*100))  

if __name__ == "__main__":
    main()