import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, fbeta_score
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import DataClean

# Derived from Pytorch documentation
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

# Derived from Pytorch documention
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7*7*64, 512)  # Input size calculated from the output of the last pooling layer
        self.fc2 = nn.Linear(512, num_classes) 

    def forward(self, x):
        x = self.maxpool1(torch.relu(self.conv1(x)))  
        x = self.maxpool2(torch.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x

# some help of chat gpt to create plots and print metrics
class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, device, dataClean):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.dataClean = dataClean

    def train(self, dataloader, epochs=1):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += y.size(0)
                correct_predictions += (predicted == y).sum().item()

            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = 100 * correct_predictions / total_predictions
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    def test(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += y.size(0)
                correct_predictions += (predicted == y).sum().item()

        test_loss = running_loss / len(dataloader)
        test_accuracy = 100 * correct_predictions / total_predictions
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    def analyze_results(self, dataloader):
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X)
                all_preds.extend(preds.argmax(1).cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f2_score = fbeta_score(all_targets, all_preds, beta=2, average=None)
        worst_f2_index = np.argmin(f2_score)
        worst_f2_label = self.dataClean.label_names[str(worst_f2_index)]

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision.mean():.4f}, Recall: {recall.mean():.4f}, Mean F2 Score: {f2_score.mean():.4f}")
        print(f"Worst F2 Score Label: {worst_f2_label}, F2 Score: {f2_score[worst_f2_index]:.4f}")


        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        plt.show()
    