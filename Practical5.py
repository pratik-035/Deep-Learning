# Practical 5 : Implement Convolutional Neural Network CNN for Digital Recognition on the MNIST Dataset

import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow.keras.models 


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the pixel value between 0 and 1 

X_train = X_train.astype('float32') / 255.0 
X_test = X_test.astype('float32') / 255.0 
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Define CNN architecture 

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'), 
    keras.layers.Dense(10, activation='softmax')
])

# compile the model 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train 

history = model.fit(X_train, y_train, epochs=15, batch_size=128, validation_data=(X_test, y_test))

# evaluate 
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy : ", test_acc)

# show prediction for a simple input image 

sample_img = X_test[0]
sample_label = y_test[0]
sample_img = np.expand_dims(sample_img, 0)

pred = model.predict(sample_img)
pred_label = np.argmax(pred)

print("Sample image true label : ", sample_label)
print("Sample image predicted label : ", pred_label)

plt.imshow(sample_img.squeeze(), cmap='gray')
plt.show()