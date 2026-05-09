import torch                                   # main module
from torch import nn                           # we'll use neural networks...
from torch.utils.data import DataLoader        # DataLoader is a convenient wrapper (see below)
from torchvision import datasets               # we'll use a "vision" dataset (for an image classification task)
from torchvision.transforms import ToTensor
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 128)
        self.readout = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.readout(x)
        return x
    

model_seq = NeuralNetwork().to("cuda")
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_seq.parameters(), lr=0.01)


def train(dataloader, model, loss_fn, optimizer):
    size = dataloader.dataset
    num_batches = len(dataloader)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to("cuda"), y.to("cuda")
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):
    size = dataloader.dataset
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in enumerate(dataloader    ):
            X, y = X.to("cuda"), y.to("cuda")
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size