import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Dataset:
    def __init__(self):
        self.train_path = "C:/Users/Ferna/AIAssignments/Task3/train.csv"
        self.test_path = "C:/Users/Ferna/AIAssignments/Task3/test.csv"

    def load_data(self, path):
        return pd.read_csv(path)

    def normalize_data(self, data):
        data.iloc[:, 1:] = data.iloc[:, 1:] / 255.0
        return data

    def split_data(self, data):
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        return X, y

    def get_train_data(self):
        train_data = self.load_data(self.train_path)
        train_data = self.normalize_data(train_data)
        X, y = self.split_data(train_data)
        return X, y

    def get_test_data(self):
        test_data = self.load_data(self.test_path)
        test_data = self.normalize_data(test_data)
        X, y = self.split_data(test_data)
        return X, y
    
    def print_random_image(self):
        train_data = self.load_data(self.train_path)
        train_data = self.normalize_data(train_data)
        random_index = np.random.randint(train_data.shape[0])
        image_size = 28
        image = train_data.iloc[random_index, 1:].values.reshape(image_size, image_size)
        label = train_data.iloc[random_index, 0]
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')
        plt.show()
    