import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

class Dataset:
    def __init__(self):
        self.train_path = "C:/Users/Ferna/AIAssignments/train.csv"
        self.test_path = "C:/Users/Ferna/AIAssignments/test.csv"
        self.label_names = self.load_label_names()

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
    
    def load_label_names(self):
        # Load label names from config.json
        with open("C:/Users/Ferna/AIAssignments/Task3/config.json", "r") as file:
            label_names = json.load(file)
        return label_names
    
    def print_random_image(self):
        train_data_pic = self.load_data(self.train_path)
        random_index = np.random.randint(train_data_pic.shape[0])
        image_size = 28
        image = train_data_pic.iloc[random_index, 1:].values.reshape(image_size, image_size)
        label_index = train_data_pic.iloc[random_index, 0]
        label_name = self.label_names[str(label_index)]
        plt.imshow(image, cmap='gray', interpolation='nearest')
        plt.title(f'Label: {label_name}')
        plt.axis('off')
        plt.show()
    