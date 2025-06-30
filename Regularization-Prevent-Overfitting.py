"""
Practical 3 : Program to implement regularization to prevent the model from overfitting    
"""

import tensorflow as tf 
import matplotlib.pyplot as plt

((train_data, train_labels), (test_data, test_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data.reshape((60000, 784)) / 255.0 
test_data = test_data.reshape((10000, 784)) / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels) 
test_labels = tf.keras.utils.to_categorical(test_labels)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784, ), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), 
    
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    loss = 'categorical_crossentropy', 
    metrics=['accuracy']
)

history = model.fit(train_data, train_labels, epochs=50, batch_size=128, validation_data=(test_data, test_labels))


plt.figure(figsize=(12, 5))

# Plot the training loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# Plot the training accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()