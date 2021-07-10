# TRURAPID COVID-19 Detection Server

This repository is a Python project for TRURAPID COVID strip test detection and classification. The main idea of the project is that it is a flask server that receives input from a web page, then passes this input to two deep learning models to be detected then classified and the final output is returned back to be printed on the web page.

## Installation of Dependencies

 1. `cd` to the **covid_detection_server** directory
 2. `pip install -r requirements.txt` to install the needed python libraries for the project
 
## Downloading Model Files

 1. Go to this [Google Drive link](https://drive.google.com/drive/u/0/folders/1hWAwd_s6g3m4eFaEEzv058YhECaPWSsQ) and download all the models in this folder and place them into the `covid_detection_server/models` directory.

## Running and Inference

 1. Give permissions the main shell script to be executable by `chmod +x run_server.sh`
 2. Run by `./run_server.sh`

## Models Training
~COMING SOON