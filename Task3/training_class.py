import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, fbeta_score
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output

from dataset import DataClean

# Derived from Pytorch documentation
class setUp(DataClean):
    def __init__(self, images, labels, transform=None):
        self.X = images  # Image data
        self.y = labels  # Labels for the images
        self.transform = transform  # Transformations to be applied on images (e.g., normalization)

    def __len__(self):
        # Returns the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieves an item by index, applying transformations if any
        image = self.X.iloc[idx].values.astype(np.float32).reshape(28, 28)  # Convert the image into a 28x28 float32 array
        if self.transform:
            image = self.transform(image)  # Apply transformations if defined
        label = self.y.iloc[idx]  # Get the label for the corresponding image
        return image, label  # Return the processed image and its label

# Derived from Pytorch documention
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        # First pooling layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        # Second pooling layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # First fully connected layer
        self.fc1 = nn.Linear(7*7*64, 512)  # Input size calculated from the output of the last pooling layer
        # Output layer
        self.fc2 = nn.Linear(512, num_classes)  # Maps the features to class scores

    def forward(self, x):
        # Defines the forward pass of the network
        x = self.maxpool1(torch.relu(self.conv1(x)))  # Apply conv1 -> ReLU -> maxpool1
        x = self.maxpool2(torch.relu(self.conv2(x)))  # Apply conv2 -> ReLU -> maxpool2
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = torch.relu(self.fc1(x))  # Apply fc1 -> ReLU
        x = self.fc2(x)  # Compute the final class scores
        return x

# Model training and evaluation class
class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        self.model = model  # The neural network model
        self.loss_fn = loss_fn  # Loss function for training
        self.optimizer = optimizer  # Optimizer for training
        self.device = device  # Computing device (CPU or GPU)
        self.history = {'accuracy': [], 'precision': [], 'recall': []}
    
    def calculate_metrics(self, pred, targets):
        # Calculate accuracy, precision, and recall metrics
        pred = torch.argmax(pred, dim=1).numpy()  # Get the predicted classes
        targets = targets.numpy()  # Convert targets to numpy for metric calculation
        accuracy = accuracy_score(targets, pred)
        precision = precision_score(targets, pred, average='macro')
        recall = recall_score(targets, pred, average='macro')
        return accuracy, precision, recall

    def train(self, dataloader):
        # Train the model for one epoch
        size = len(dataloader.dataset)
        self.model.train()  # Set model to training mode
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)  # Move data to the specified device
            pred = self.model(X)  # Make predictions
            loss = self.loss_fn(pred, y)  # Compute loss

            self.optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            self.optimizer.step()  # Update model parameters

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader):
        # Evaluate the model's performance on a dataset
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()  # Set model to evaluation mode
        test_loss, correct = 0, 0
        with torch.no_grad():  # Disable gradient computation for evaluation
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)  # Move data to the specified device
                pred = self.model(X)  # Make predictions
                test_loss += self.loss_fn(pred, y).item()  # Accumulate the loss
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # Count correct predictions

        test_loss /= num_batches  # Calculate average loss
        correct /= size  # Calculate accuracy
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    def analyze_results(self, dataloader):
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X)
                all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average=None)  # Precision for each class
        recall = recall_score(all_targets, all_preds, average=None)  # Recall for each class
        f2_score = fbeta_score(all_targets, all_preds, beta=2, average=None)  # F2 score for each class
        
        # Identify the class with the lowest F2 score
        lowest_f2_index = np.argmin(f2_score)
        print(f"Lowest F2 Score Class: {lowest_f2_index}, F2 Score: {f2_score[lowest_f2_index]}")
        
        # Plot the confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()

        