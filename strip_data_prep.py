# File: strip_data_prep.py
# Author: @MichaelHannalla
# Project: Trurapid COVID-19 Strips Detection Server with Python
# Description: Python file for preparing the small strip crop dataset to train the classifier

from detecto.core import Model
from detecto.utils import read_image
from detecto.visualize import plot_prediction_grid
from utils import get_crop
import matplotlib.pyplot as plt
import cv2
import os


def main():

    data_path = "data/train"
    crop_path = "data/crops"
    data = os.listdir(data_path)
    model = Model.load('models/strip_detector_weights_pass2.pth', ['strip'])

    for idx, train_img_path in enumerate(data):
        if train_img_path.endswith('.jpeg') or train_img_path.endswith('.jpg'):
            train_img_full_path = data_path + "/" + train_img_path
            train_img = read_image(train_img_full_path)
            labels, boxes, scores = model.predict(train_img)
            cropped = get_crop(train_img, boxes[0])
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            write_path = os.path.join(crop_path, "crop{}.jpg".format(idx))
            write_success = cv2.imwrite(write_path, cropped)
            if write_success:    
                print("Done sample {} with relative path {}".format(idx, crop_path + "/crop{}.jpeg".format(idx)))    
    
if __name__ == "__main__":
    main()
    