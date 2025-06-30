# Practical 4 : Implement DL for the prediction of the autoencoder from the test data (e.g MNIST Dataset)

import tensorflow as tf  
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt  


# Load the MNIST dataset 
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()


x_train = x_train.astype("float32") / 255.0 
x_test = x_test.astype("float32") / 255.0 

# encoder architecutre define 

encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]), 
    keras.layers.Dense(128, activation="relu"), 
    keras.layers.Dense(64, activation="relu",), 
    keras.layers.Dense(32, activation="relu",), 
])


# decoder architecture define 

decoder = keras.models.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape= [32]), 
    keras.layers.Dense(128, activation="relu"), 
    keras.layers.Dense(28 * 28, activation="sigmoid"), 
    keras.layers.Reshape([28,28])
])


# combine encoder and decoder into autoencoder model 
autoencoder = keras.models.Sequential([encoder, decoder])


# compile autoencoder model 
autoencoder.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001))


history = autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, validation_data = (x_test, x_test))


# use trained autoencoder to predict the reconstructed images for the test data 
decoded_imgs  = autoencoder.predict(x_test)


n = 5 # number of images to display 

plt.figure(figsize=(20, 4))

for i in range(n): 
    # display original images 
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


for i in range(n):
    # display reconstructed images 
    ax = plt.subplot(2,n,i+n+1)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()