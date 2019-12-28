import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio

import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.federated import utils
import settings

# This is important to exploit the GPU if it is available
use_cuda = torch.cuda.is_available()

# Seed for the random number generator
torch.manual_seed(1)

device = torch.device("cuda" if use_cuda else "cpu")

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

class TestingRemote(nn.Module):
    def __init__(self):
        super(TestingRemote, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TestingRemote2(nn.Module):
    def __init__(self):
        super(TestingRemote2, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GRUModel(nn.Module):
    # To understand the meaning of this variable visit the page of pytorch: https://pytorch.org/docs/master/nn.html#gru
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, h):
        out, h = self.gru(x,h)
        out = self.fc(self.sigmoid(out[:, -1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

# Loss function
# it needs to be serializable. 
#  We can define a usual function just changing it to use jit.
@torch.jit.script
def loss_fn(target, pred):
    return ((target.view(pred.shape).float() - pred.float()) ** 2).mean()
    # or for example you can use
    # return F.nll_loss(input=pred, target=target)

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

# TODO implements the asynchronous! Because it could require a lot of time
async def train_remote(
    worker: websocket_client.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    optimizer: str,
    max_nr_batches: int,
    epochs: int,
    lr: float,
    ):
    """Send the model to the worker and fit the model on the worker's training data.
    Args:
        worker: Remote location, where the model shall be trained.
        traced_model: Model which shall be trained.
        batch_size: Batch size of each training step.
        optimizer: name of the optimizer to be used
        max_nr_batches: If > 0, training on worker will stop at min(max_nr_batches, nr_available_batches).
        epochs: Number of epochs to perform remotely
        lr: Learning rate of each training step.
    Returns:
        A tuple containing:
            * worker_id: Union[int, str], id of the worker.
            * improved model: torch.jit.ScriptModule, model after training at the worker.
            * loss: Loss on last training batch, torch.tensor.
    """
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=epochs,
        optimizer=optimizer,
        optimizer_args={"lr": lr},
    )
    # When the training is started this remote worker can be removed from the devices to be training
    del settings.training_devices[worker.id] 
    try: 
        train_config.send(worker)
        loss = await worker.async_fit(dataset_key="training", return_ids=[0])
        model = train_config.model_ptr.get().obj
    finally:
        # After the training, close the websocket with the server
        worker.close()

    return worker.id, model, loss



def evaluate(test_loader, device):
    """Evaluate the model. This method can be used only for a local evaluation of the global model
    Args:
        model: model to evaluate
        args: parameters for the evaluation (see class Arguments)
        test_loader: loader for the data to test
        device: enable the possibility to exploit the GPU
    Returns:
        no return
    """
    print("Local evaluation start...")
    model = TestingRemote2()
    model.load_state_dict(torch.load("./test.pth"))
    model.eval()

    # Defining Loss Function and optimizer
    criterion = nn.MSELoss()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(output)
            test_loss += criterion(output, target).item()# sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
