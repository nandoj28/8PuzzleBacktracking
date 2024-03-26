# import tensorflow as tf
import keras as kers
from dataClean import DataClean  # Make sure your class is in a file named dataset_class.py

# Ensure the DataClean class uses the labels dictionary correctly as discussed previously.

def create_model(input_shape, num_classes):
    model = kers.models.Sequential([
        kers.layers.Flatten(input_shape=input_shape),
        kers.layers.Dense(128, activation='relu'),
        kers.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def main():
    # Initialize dataClean
    dataClean = DataClean()
    X_train, y_train = dataClean.get_train_data()
    X_test, y_test = dataClean.get_test_data()
    
    # Model configuration
    input_shape = (28, 28)  # MNIST Fashion images are 28x28
    num_classes = 10  # 10 different articles of clothing
    
    # Create and compile the model
    model = create_model(input_shape, num_classes)
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=2)
    
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}, Test loss: {test_loss}')

if __name__ == '__main__':
    main()
