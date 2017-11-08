from PIL import Image
import numpy as np
import nn

import pdb

image = Image.open("test.jpg")
pxls = image.load()
img_width, img_height = image.size;

print("Image size: " + str(img_width) + "x" + str(img_height))

print("Extracting pixel values...")

X = np.array([[i, j] for i in range(img_width) for j in range(img_height)])
pxl_vals = [pxls[i, j] for i in range(img_width) for j in range(img_height)]
Y_ = np.array([[px[0], px[1], px[2]] for px in pxl_vals])

print("Extracted pixel values!")

print("Configuring neural net...")
nn_config = [2, 20, 3]

print("Neural net architecture: " + str(nn_config))

nn = nn.NeuralNet(nn_config, param_delta=1e-5)

print("Training neural net...")
nn.train(X, Y_, 200)
print("Trained neural net!")

print("Evaluating...")
Y = nn.eval(X)

print("Writing new pixel values...")
for i in range(img_width):
    for j in range(img_height):
        pdb.set_trace()
        pxls[i, j] = Y[i, j]

print("Saving to file...")
image.save("test_out.jpg", "JPEG")
print("Done!")