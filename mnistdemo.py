import tensorflow as tf
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
tf.keras.layers.Flatten()
plt.imshow(x_train[0], cmap='gray')
plt.show()

print("x_train[0]:\n", x_train[0])