import pandas as pd
import ast
import torch
import numpy as np
import torch.nn as nn
import re
from torch.utils.data import DataLoader, Dataset
import cv2
import pickle
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

from attentionBiLSTM import ActionRecognizationModel


with open('train1.pkl', 'rb') as file:
    loaded_list = pickle.load(file)

# print(loaded_list)

batch_size = 1
train_loader = DataLoader(loaded_list, batch_size=batch_size, shuffle=True)
# print(loaded_data)

# Define your model, loss function, and optimizer
attentionBiLSTM_model = ActionRecognizationModel(1, 33, 16 , 2, rightArm_input=6, leftArm_input=6, trunk_input=18, rightLeg_input=6, leftLeg_input=6)
attentionBiLSTM_criterion = nn.BCEWithLogitsLoss()
attentionBiLSTM_optimizer = torch.optim.Adam(attentionBiLSTM_model.parameters(), lr=1e-1)

num_epochs = 10
for epoch in range(num_epochs):
    attentionBiLSTM_model.train()
    running_loss = 0.0

    for batch in train_loader:
        # Unpack the batch
        keypoints_sequences, bbox_sequences, labels = batch

        # Zero the gradients
        attentionBiLSTM_optimizer.zero_grad()

        #forwatd pass
        output = attentionBiLSTM_model(keypoints_sequences,bbox_sequences)

        print('output:: ',output)
        labels = torch.tensor([1.0, 0.0])

        loss = attentionBiLSTM_criterion(output, labels)

        loss.backward()

        #update the model's parameters
        attentionBiLSTM_optimizer.step()
        
        running_loss += loss.item()

    # Print the average loss for this epoch
print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss / len(train_loader)}")
