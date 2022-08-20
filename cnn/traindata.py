import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import datasets


cifar10 = datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
plt.imshow(x_train[0])
plt.show()
print("x_train[0]:\n", x_train[0])
print("y_train[0]:\n", y_train[0])
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)