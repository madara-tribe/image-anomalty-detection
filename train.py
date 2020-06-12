import time
from keras import Input
from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential, Model, model_from_json
import matplotlib.pyplot as plt
import keras.backend as K
import os
import math
import numpy as np
import cv2
from keras.preprocessing import image
from functools import partial
from random import random
from math import floor

from styleGAN.loss import gradient_penalty_loss
from styleGAN.AdaIN import AdaInstanceNormalization
from styleGAN.styleGAN import styleGAN


# Style Z
def noise(batch_size):
    return np.random.normal(0.0, 1.0, size = [batch_size, latent_size])

# Noise Sample
def noiseImage(batch_size):
    return np.random.uniform(0.0, 1.0, size = [batch_size, im_size, im_size, 1])


# load image
def load_dataset(img_path):
  img=[]
  for imgs in os.listdir(img_path):
    imgs = cv2.imread(img_path+'/'+imgs)
    if imgs is not None:
      imgs = cv2.resize(imgs, (256, 256))
      img.append(imgs/255)
  inputs=np.array(img)
  return inputs

  

def train(weights=None):
    # configure
    im_size = 256
    latent_size = 512

    GAN = styleGAN()
    G = GAN.generator()
    D = GAN.discriminator()
    if weights is True:
        G.load_weights("styleGAN/tb/generator.h5")
        D.load_weights("styleGAN/tb/discriminator.h5")

    # create dismodel (G doesn't update but D does)
    D.trainable = True
    for layer in D.layers:
        layer.trainable = True

    G.trainable = False
    for layer in G.layers:
        layer.trainable = False

    # Real Pipeline
    ri = Input(shape = [im_size, im_size, 3])
    dr = D(ri)
    # Fake Pipeline
    gi = Input(shape = [latent_size])
    gi2 = Input(shape = [im_size, im_size, 1])
    gi3 = Input(shape = [1])
    df = D(G([gi, gi2, gi3]))
    # Samples for gradient penalty
    da = D(ri)

    disgan = Model(inputs=[ri, gi, gi2, gi3], outputs=[dr, df, da])
    # Create partial of gradient penalty loss
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = 5)
    disgan.compile(optimizer=Adam(lr=0.0002, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss=['mse', 'mse', partial_gp_loss])


    # create admodel (G does update but D doesn't)
    D.trainable = False
    for layer in D.layers:
        layer.trainable = False

    G.trainable = True
    for layer in G.layers:
        layer.trainable = True

    gi = Input(shape = [latent_size])
    gi2 = Input(shape = [im_size, im_size, 1])
    gi3 = Input(shape = [1])
    df = D(G([gi, gi2, gi3]))
    adgan = Model(inputs = [gi, gi2, gi3], outputs = df)

    adgan.compile(optimizer = Adam(lr=0.0001, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss = 'mse')

    adgan.summary()
    disgan.summary()



    x_train = load_dataset('image_folder')
    print(x_train.shape)

    """ TRAIN """
    epochs = 1000
    batch_size = 10
    counter=1
    save_dir = 'logss/'

    # Start training loop

    for epoch in range(epochs):
        batch_idx= min(len(x_train), np.inf) // batch_size
        for idx in range (0, batch_idx):
            # train disgan
            real_images  = x_train[idx*batch_size:(idx+1)*batch_size]
            train_data = [real_images, noise(batch_size), noiseImage(batch_size), np.ones((batch_size, 1))]
            d_loss = disgan.train_on_batch(train_data, [np.ones((batch_size, 1)),
                                                        -np.ones((batch_size, 1)), np.ones((batch_size, 1))])

            # train adgan
            g_loss = adgan.train_on_batch([noise(batch_size), noiseImage(batch_size), np.ones((batch_size, 1))], 
                                            np.zeros((batch_size, 1), dtype=np.float32))
        
            if np.mod(counter, 10)==1:
                print("Epoch: [%2d] [%4d/%4d], g_loss: %s, d_loss: %s" % (epoch, idx, batch_idx, g_loss, d_loss))
            counter += 1


            if np.mod(counter, 100)==1:
                # Save model weights
                G.save_weights('styleGAN/tb/generator.h5')
                D.save_weights('styleGAN/tb/discriminator.h5')

                print('discriminator loss at step %s: %s' % (epoch, d_loss))
                print('adversarial loss at step %s: %s' % (epoch, g_loss))

                # plot for evaluation
                n = noise(batch_size)
                n2 = noiseImage(batch_size)
                im2 = G.predict([n, n2, np.ones([batch_size, 1])])
                r12 = np.concatenate(im2[:4], axis = 1)
                c1 = np.concatenate([r12], axis = 0)

                img1 = image.array_to_img(c1 * 255., scale=False)
                img2 = image.array_to_img(im2[0] * 255., scale=False)
                plt.imshow(img1)
                plt.show()

                plt.imshow(img2)
                plt.show()

                img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))


if __name__ == '__main__':
    train()