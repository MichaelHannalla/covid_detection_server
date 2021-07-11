# File: strip_classifier_train.py
# Author: @MichaelHannalla
# Project: Trurapid COVID-19 Strips Detection Server with Python
# Description: Python file for training the PyTorch strip classifier model

import torch
import torch.nn.functional as F

from torch import nn, optim
from utils import strip_dataloader, input_layer_dim

def main():

    # Prepare the dataset
    train_data_path = "data/crops/train"
    test_data_path = "data/crops/test"
    trainloader = strip_dataloader(train_data_path)
    testloader = strip_dataloader(test_data_path)

    # Define the loss
    criterion = nn.NLLLoss()

    # Define the neural network architecture
    model = nn.Sequential(
                      nn.Linear(input_layer_dim, 512),
                      nn.ReLU(),
                      nn.Linear(512, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 2),
                      nn.LogSoftmax(dim = 1)
                     )

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    # Define the epochs
    epochs = 1000

    train_losses, test_losses = [], []

    for e in range(epochs):
        running_loss = 0

        images, labels = trainloader
        # Flatten images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        # Training pass
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computation
        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()
            
            # Validation pass
            images, labels = testloader
            images = images.view(images.shape[0], -1)
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        model.train()
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        
        print("Epoch: {}/{}..".format(e+1, epochs),
            "Training loss: {:.3f}..".format(running_loss/len(trainloader)),
            "Test loss: {:.3f}..".format(test_loss/len(testloader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

    
    torch.save(model, "models/strip_classifier_mini.pth")

if __name__ == "__main__":
    main()
