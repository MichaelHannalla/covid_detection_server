# File: strip_detector_train.py
# Author: @MichaelHannalla
# Project: Trurapid COVID-19 Strips Detection Server with Python
# Description: Python file for training the strip detector based on detecto on a single input image


from detecto.core import Dataset, Model

def main():

    #TODO: using the argparser
    train_dataset = Dataset('data/train')
    model = Model(['strip'])
    val_dataset = Dataset('data/test')
    losses = model.fit(train_dataset, val_dataset, epochs=200, learning_rate=0.01,
                   gamma=0.2, lr_step_size=5, verbose=True)

    
    model.save('models/covid_strip_weights_single_class.pth')

if __name__ == "__main__":
    main()
