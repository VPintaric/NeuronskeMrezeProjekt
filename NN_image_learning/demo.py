import numpy as np
import pyglet as pg
import nn

from helpers import *
from pyglet.gl import *

import pdb

class DemoWindow(pg.window.Window):
    def __init__(self, width, height, img, fullscreen=False):
        pg.window.Window.__init__(self, width=width, height=height, fullscreen=fullscreen)
        self.img = img

    def on_draw(self):
        self.clear()
        texture = self.img.get_texture()
        #glEnable(texture.target)
        #glBindTexture(texture.target, texture.id)
        #glTexParameteri(texture.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        self.img.blit(0, 0, width=self.width, height=self.height)

if __name__ == '__main__':
    W, H = 700, 700              # window dimensions
    FULLSCREEN = False           # or fullscreen, maybe?
    FPS = 60.                    # maximum frames per second
    IMAGE_FILE = "space.jpg"     # image to compress/train neural net on
    ITERATIONS_PER_FRAME = 25    # how many neural net training steps to do per frame 
    ARCH = [2, 10, 20, 20, 3]     # neural net architecture configuration
    PARAM_DELTA = 1e-2            # learning rate for adam optimizer

    img = pg.image.load(IMAGE_FILE).get_image_data()
    img_w, img_h = img.width, img.height
    pixels = img.get_data("RGB", img_w * 3)

    X = [[i, j] for i in range(img_w) for j in range(img_h)]
    pxl_arr = pixel_bytes_to_array(pixels, img_w, img_h)

    trained_pxl_arr = np.zeros_like(pxl_arr)
    trained_pixels = array_to_pixel_bytes(trained_pxl_arr)
    trained_img = pg.image.ImageData(img_w, img_h, "RGB", trained_pixels)

    net = nn.NeuralNet(ARCH, PARAM_DELTA)

    window = DemoWindow(W, H, trained_img, FULLSCREEN)
    def update(dt):
        global trained_img, img_w, img_h, net, pxl_arr
        for i in range(ITERATIONS_PER_FRAME):
            loss = net.train_iteration(X, pxl_arr)
        print("loss = %lf" % loss)
        trained_pxl_arr = net.eval(X)
        trained_pixels = array_to_pixel_bytes(trained_pxl_arr)
        trained_img.set_data("RGB", img_w * 3, trained_pixels)


    pg.clock.schedule_interval(update, 1. / FPS)
    pg.app.run()
