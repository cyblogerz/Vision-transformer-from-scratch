"""
Contains functions for training and testing the model
"""

import torch

from tqdm.auto import tqdm
from typing import List, Tuple,Dict

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a pytorch model for a single epoch"""

    #put model in train mode
    model.train()

    #setup train loss and accuracy values
    train_loss,train_acc = 0,0

    #loop through data loader data batches
    for batch, (X,y) in enumerate(dataloader):
        #send data to target device
        X, y = X.to(device), y.to(device)

        #Forward pass
        y_pred = model(X)

        #Calcualte and accumalte loss
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()

        #Optimizer zero grad
        optimizer.zero_grad()

        #loss backward
        loss.backward()

        #optimizer step
        optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)


          # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        
        return train_loss, train_acc






