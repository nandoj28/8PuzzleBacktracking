import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from torchvision import transforms

from dataset import DataClean

class setUp(DataClean):
    def __init__(self, images, labels, transform=None):
        self.X = images
        self.y = labels
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        image = self.X.iloc[idx].values.astype(np.float32).reshape(28, 28)
        if self.transform:
            image = self.transform(image)
        label = self.y.iloc[idx]
        return image, label

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)  # Increased from 1 to 2
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=4)  # Increased from 2 to 4
        self.fc1 = nn.Linear(256, 64)  # Reduced from 128 to 64
        self.fc2 = nn.Linear(64, num_classes)  # Reduced from 128 to 64
        
    def forward(self, x):
        x = self.maxpool1(torch.relu(self.conv1(x)))
        
        # Since we removed conv2 and maxpool2, we directly proceed to flattening and the dense layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_metrics(pred, targets):
    pred = torch.argmax(pred, dim=1).numpy()
    targets = targets.numpy()
    accuracy = accuracy_score(targets, pred)
    precision = precision_score(targets, pred, average='macro')
    recall = recall_score(targets, pred, average='macro')
    return accuracy, precision, recall

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Load and preprocess data
fashion_mnist = DataClean()
X_train, y_train = fashion_mnist.get_train_data()
X_test, y_test = fashion_mnist.get_test_data()

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Adjustments for using the transform
train_dataset = setUp(X_train, y_train, transform=transform)
val_dataset = setUp(X_val, y_val, transform=transform)
test_dataset = setUp(X_test, y_test, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleCNN(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(val_dataloader, model, loss_fn)
print("Done!")

# Testing the model
test(test_dataloader, model, loss_fn)
