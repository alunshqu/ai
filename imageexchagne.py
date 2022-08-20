import numpy as np
from PIL import Image

def printArr(imgarr:[]):
    for i in range(28):
        for j in range(28):
            print(str(img_arr[i][j]), end='\t')
        print("\n")

image_path = "8.png"
img = Image.open(image_path)
img = img.resize((28, 28))
img_arr = np.array(img.convert("L"))
printArr(img_arr)

for i in range(28):
    for j in range(28):
        if img_arr[i][j] > 150:
            img_arr[i][j] = 0
        else:
            img_arr[i][j] = 255

printArr(img_arr)

