from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models

model_save_path = "D:\\aidemo\\mnist.ckpt"
model = models.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.load_weights(model_save_path)
image_path = "8.png"
img = Image.open(image_path)
# img.show("orig")
img = img.resize((28, 28))
# img.show("target")
img_arr = np.array(img.convert("L"))
for i in range(28):
    for j in range(28):
        if img_arr[i][j] > 140:
            img_arr[i][j] = 0
        else:
            img_arr[i][j] = 255

for i in range(28):
    for j in range(28):
        print(str(img_arr[i][j]), end='\t')
    print("\n")

img_arr = img_arr / 255.0
x_predict = img_arr[tf.newaxis, ...]
result = model.predict(x_predict)
pred = tf.argmax(result, axis=1)
print("\n")
tf.print(pred)