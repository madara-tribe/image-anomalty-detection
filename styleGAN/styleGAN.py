from keras import backend as K
from keras import Input
from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential, Model
import os
import pandas as pd
import numpy as np
import cv2
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from AdaIN import AdaInstanceNormalization

im_size = 256
latent_size = 512

def g_block(inp, style, noise, fil, u = True):

    b = Dense(fil)(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil)(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)

    if u:
        out = UpSampling2D(interpolation = 'bilinear')(inp)
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    else:
        out = Activation('linear')(inp)

    out = AdaInstanceNormalization()([out, b, g])
    out = add([out, n])
    out = LeakyReLU(0.01)(out)

    b = Dense(fil)(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil)(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = AdaInstanceNormalization()([out, b, g])
    out = add([out, n])
    out = LeakyReLU(0.01)(out)

    return out


def d_block(inp, fil, p = True):
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
    route2 = LeakyReLU(0.01)(route2)
    if p:
        route2 = AveragePooling2D()(route2)
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(route2)
    out = LeakyReLU(0.01)(route2)

    return out



class styleGAN(object):

    def discriminator(self):

        inp = Input(shape = [im_size, im_size, 3])

        # Size
        x = d_block(inp, 16) #Size / 2
        x = d_block(x, 32) #Size / 4
        x = d_block(x, 64) #Size / 8

        if (im_size > 32):
            x = d_block(x, 128) #Size / 16

        if (im_size > 64):
            x = d_block(x, 192) #Size / 32

        if (im_size > 128):
            x = d_block(x, 256) #Size / 64

        if (im_size > 256):
            x = d_block(x, 384) #Size / 128

        if (im_size > 512):
            x = d_block(x, 512) #Size / 256


        x = Flatten()(x)

        x = Dense(128)(x)
        x = Activation('relu')(x)

        x = Dropout(0.6)(x)
        x = Dense(1)(x)

        return Model(inputs = inp, outputs = x)


    def generator(self):

        #Style FC, I only used 2 fully connected layers instead of 8 for faster training
        inp_s = Input(shape = [latent_size])
        sty = Dense(512, kernel_initializer = 'he_normal')(inp_s)
        sty = LeakyReLU(0.1)(sty)
        sty = Dense(512, kernel_initializer = 'he_normal')(sty)
        sty = LeakyReLU(0.1)(sty)

        #Get the noise image and crop for each size
        inp_n = Input(shape = [im_size, im_size, 1])
        noi = [Activation('linear')(inp_n)]
        curr_size = im_size
        while curr_size > 4:
            curr_size = int(curr_size / 2)
            noi.append(Cropping2D(int(curr_size/2))(noi[-1]))

        #Here do the actual generation stuff
        inp = Input(shape = [1])
        x = Dense(4 * 4 * 512, kernel_initializer = 'he_normal')(inp)
        x = Reshape([4, 4, 512])(x)
        x = g_block(x, sty, noi[-1], 512, u=False)

        if(im_size >= 1024):
            x = g_block(x, sty, noi[7], 512) # Size / 64
        if(im_size >= 512):
            x = g_block(x, sty, noi[6], 384) # Size / 64
        if(im_size >= 256):
            x = g_block(x, sty, noi[5], 256) # Size / 32
        if(im_size >= 128):
            x = g_block(x, sty, noi[4], 192) # Size / 16
        if(im_size >= 64):
            x = g_block(x, sty, noi[3], 128) # Size / 8

        x = g_block(x, sty, noi[2], 64) # Size / 4
        x = g_block(x, sty, noi[1], 32) # Size / 2
        x = g_block(x, sty, noi[0], 16) # Size

        x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(x)
        return Model(inputs = [inp_s, inp_n, inp], outputs = x)
