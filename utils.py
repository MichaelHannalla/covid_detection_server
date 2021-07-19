# File: utils.py
# Author: @MichaelHannalla
# Project: Trurapid COVID-19 Strips Detection Server with Python
# Description: Python file for utilities and helper functions used across the whole project

import os
import cv2
import numpy as np
import torch
from PIL import Image

from scipy.signal import savgol_filter

#TODO: using config files instead of variables
labels = ['strip']
crop_size = [784, 256]
input_layer_dim = crop_size[1] * crop_size[0] * 3
positive = 0
negative = 1
savgol_window = 25
grad_thresh = 0.5


def get_image_data(uploaded_file):
    '''
        Function to convert request file to OpenCV compatible format
        Inputs:
            uploaded_file : Uploaded file request with flask
        Output:
            pil_image : image object (RGB)
    '''
    pil_image = Image.open(uploaded_file)
    return np.array(pil_image)

def get_crop(img, box):
    '''
        A cropping function that performs getting only a certain portion of an image
        Inputs:
            img (string): whole image
            box (tensor): torch tensor representing bounding box locations
        Output:
            crop : desired area of interest
    '''
    xmin, ymin, xmax, ymax = box.numpy().astype(np.uint64)
    crop = img[ymin:ymax, xmin:xmax, :]
    return crop

def negate_image(img):
    '''
        Function to return the negative of an image
    '''
    neg = 255 - img
    return neg

def strip_dataloader(path):
    '''
        Dataloader function for loading crops from the training folder to train the strip classifier model
        Inputs:
            path (string): path to the crops folder
        Output:
            data_tensor, label_tensor: two pytorch tensors for training
    '''
    data = os.listdir(path)
    data_vec = np.array([]).reshape(0, 3, crop_size[0], crop_size[1])
    labels_vec = []
    for idx, datapoint_path in enumerate(data):
        datapoint_label = -np.inf # implausible value to raise errors if not modified
        if datapoint_path.startswith('p') and (datapoint_path.endswith('.jpeg') or datapoint_path.endswith('.jpg')):
            datapoint_label = positive
        if datapoint_path.startswith('n') and (datapoint_path.endswith('.jpeg') or datapoint_path.endswith('.jpg')):
            datapoint_label = negative

        img_full_path = path + "/" + datapoint_path
        img = cv2.imread(img_full_path)

        img = cv2.resize(img, (crop_size[1], crop_size[0]))
        img = img.reshape(1, 3, crop_size[0], crop_size[1])

        data_vec = np.vstack((data_vec, img))
        labels_vec.append(datapoint_label)


    data_tensor = torch.from_numpy(data_vec.astype(np.float32))
    labels_tensor = torch.from_numpy(np.array(labels_vec))
                    
    return data_tensor, labels_tensor

def get_image_tensor(img):
    '''
        Converter function from numpy image to torch tensor with compatible dimensions
        Inputs:
            img : image as numpy array
        Output:
            img_tensor : image as a torch tensor
    '''
    img = cv2.resize(img, (crop_size[1], crop_size[0]))
    img = img.reshape(1, 3, crop_size[0], crop_size[1])
    img_tensor = torch.from_numpy(img.astype(np.float32))
    img_tensor = img_tensor.view(img_tensor.shape[0], -1)
    return img_tensor

def get_label_from_onehot(onehot_tensor):
    '''
        Converts from one-hot encoding to sparse encoding
    '''
    onehot_np = onehot_tensor.numpy().reshape(-1)
    label = np.argmax(onehot_np)
    return label

def classify_crop(crop_img, model):
    '''
        Main function to classify crop (desired area of a test sample)
        Inputs:
            crop_img : image as numpy array
            model : PyTorch deep learning model (classifier)
        Output:
            predicted_label : label in sparse encoding
    '''
    crop_tensor = get_image_tensor(crop_img)
    with torch.no_grad():
        result = model(crop_tensor)
    predicted_label = get_label_from_onehot(result)
    return predicted_label

def classical_classify_crop(crop_img):
    '''
        Suplementary function to classify crop (desired area of a test sample)
        Inputs:
            crop_img : image as numpy array
        Output:
            predicted_label : label in sparse encoding
    '''
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)     # convert to grayscale
    crop_img = negate_image(crop_img)                       
    intensities_vec = crop_img[:, crop_img.shape[1]//2]              
    intensities_vec = savgol_filter(intensities_vec, savgol_window, 2)     # second order savistky-golay filter with 25 window size
    grad_vec = np.gradient(intensities_vec)
    grad_vec[np.absolute(grad_vec) < grad_thresh] = 0
    grad_positive_part_of_interest = grad_vec[len(grad_vec)//2: int(3 * len(grad_vec)//4)]
    if np.any(np.absolute(grad_positive_part_of_interest) > 0):
        return positive
    else:
        return negative

def get_strip_crop(img, model):
    '''
        Main function to detect crop area (desired area of a test sample)
        Inputs:
            img : image as numpy array
            model : PyTorch deep learning model (detector)
        Output:
            cropped : area of interest
    '''
    labels, boxes, scores = model.predict(img)
    cropped = get_crop(img, boxes[0])
    return cropped

def null_func(none):
    '''
        Null function for OpenCV functions
    '''
    pass