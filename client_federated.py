import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.federated import utils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# TODO define training local workers
def train_local(worker, model, opt, epochs, federated_train_loader, args):
    # In this case the location of the worker is directly in the data
    """Send the model to the worker and fit the model on the worker's training data.
    Args:
        worker: train the model on that worker
        model: Model which shall be trained.
        opt: Optimization algorithm
        epochs: Number of epochs
        federated_train_loader: loader of the data distributed by us among the network
        args: value of the singleton class Arguments

    Returns:
        A tuple containing:
            * improved model: model after training at the worker.
            * loss: Loss on last training batch, torch.tensor.
    """
    model.train()
    result_models = {}
    for epoch in range(epochs):
        print("Training for worker: " + worker)
        for batch_idx, (data, target) in enumerate(federated_train_loader): # now it is a distributed dataset
            #print(data.location.id)
            if data.location.id == worker:
                model.send(data.location)

                # 1) Erase the previous gradients
                opt.zero_grad()

                # 2) Make a prediction
                pred = model(data)

                # 3) Calculate how much we missed
                loss = F.nll_loss(pred, target)

                # 4) figure out which weights caused us to miss
                loss.backward()

                # 5) change those weights
                opt.step()

                model.get()
                if batch_idx % args.log_interval == 0:
                    loss = loss.get() # <-- NEW: get the loss back
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                        100. * batch_idx / len(federated_train_loader), loss.item()))
    

    return model, loss



    print()

# TODO define training remote workers: websocket
def train_remote():
    print()

# TODO define testing: The testing actually could be the same for local purposes and remote purposes
def evaluate_local(model, args, test_loader, device):
    """Evaluate the model locally.
    Args:
        model: model to evaluate
        args: parameters for the evaluation (see class Arguments)
        test_loader: loader for the data to test
        device: enable the possibility to exploit the GPU
    Returns:
        no return
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
