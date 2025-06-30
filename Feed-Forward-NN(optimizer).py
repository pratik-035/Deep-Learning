# Practical 1 : Implement Feed Forward NN and train the network with different optimizers and compare

import tensorflow as tf  
import numpy as np  
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# Load the dataset 
iris = load_iris()


# Get feature and output 
X = iris.data 
y = iris.target 


# One-Hot encode labels 
lb = LabelBinarizer()
y = lb.fit_transform(y)


# Split the data ubto training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the model architecture 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4, )),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])



# Compile model with different optimizers 

optimizers = ['adam', 'sgd', 'rmsprop']

for optimizer in optimizers:
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    
    # Train the model 
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, verbose=1) 
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('Optimizer : ', optimizer)
    print('Test loss : ', loss)
    print('Test accuracy : ', accuracy)