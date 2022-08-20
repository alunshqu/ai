import tensorflow as tf
import numpy

print(tf.__version__)

x = [[1.]]
m = tf.matmul(x, x)
print(m)

a = tf.constant([1, 5],dtype=tf.int64)
print(a)


