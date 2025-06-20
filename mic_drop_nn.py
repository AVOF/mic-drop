import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.model_selection import train_test_split

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
  # Input layer (2 features -- the complex resonant frequency)
  # output (droplet radius)
  
  def __init__(self, in_features=2, h1=20, h2=20, out_features=1):
    super().__init__() # instantiate our nn.Module
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x

def run_torch(X, y, LR, n_epochs, seed_num):
    

    # Pick a manual seed for randomization
    torch.manual_seed(seed_num)
    # Create an instance of model
    model = Model()
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_num, stratify=y)
    
    # Convert X features to float tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
         
    # Convert y labels to tensors long
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)
    
    # Set the criterion of model to measure the error, how far off the predictions are from the data
    criterion = nn.MSELoss()
    
    # Use Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train model
    losses = []
    for i in range(n_epochs):
        # Go forward and get a prediction
        y_pred = model.forward(X_train) # Get predicted results
          
        # Measure the loss/error, gonna be high at first
        loss = criterion(y_pred, y_train) # predicted values vs the y_train
          
        # Keep Track of our losses
        losses.append(loss.detach().numpy())
          
        # Do some back propagation: take the error rate of forward propagation and feed it back
        # thru the network to fine tune the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate Model on Test Data Set (validate model on test set)
    with torch.no_grad(): 
        y_eval = model.forward(X_test) 
        yev = (3 * np.array(y_eval/100).flatten() / 4 / np.pi) ** (1/3)
        yte = (3 * np.array(y_test/100).flatten() / 4 / np.pi) ** (1/3)

    return X_test, yte, yev
      
# load data
n_rep = 20
x = np.loadtxt("complete_drop_data.csv", delimiter=",")
y = np.repeat(np.arange(30, 110, 5), n_rep) # ground-truth volumes

# set nn parameters
n_epochs = 5000
LR = 0.002

# run training and save test results for comparison
Xt, yte, yev = run_torch(x, y, LR, n_epochs, 8)
