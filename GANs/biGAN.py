from __future__ import print_function, division
import time
from keras.datasets import mnist
from keras import Input
from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential, Model, model_from_json
import matplotlib.pyplot as plt
import keras.backend as K
import os
import numpy as np
import cv2
from keras.preprocessing import image

from random import random
from PIL import Image
from math import floor

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
# https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py


img_shape = (28, 28, 1)
latent_dim = 100

class biGAN:
  def encoder(self):
        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(latent_dim))

        img = Input(shape=img_shape)
        z = model(img)

        return Model(img, z)

  def generator(self):
      model = Sequential()

      model.add(Dense(512, input_dim=latent_dim))
      model.add(LeakyReLU(alpha=0.2))
      model.add(BatchNormalization(momentum=0.8))
      model.add(Dense(512))
      model.add(LeakyReLU(alpha=0.2))
      model.add(BatchNormalization(momentum=0.8))
      model.add(Dense(np.prod(img_shape), activation='tanh'))
      model.add(Reshape(img_shape))
      model.summary()
      
      z = Input(shape=(latent_dim,))
      gen_img = model(z)

      return Model(z, gen_img)

  def discriminator(self):

      z = Input(shape=(latent_dim, ))
      img = Input(shape=img_shape)
      d_in = concatenate([z, Flatten()(img)])

      model = Dense(1024)(d_in)
      model = LeakyReLU(alpha=0.2)(model)
      model = Dropout(0.5)(model)
      model = Dense(1024)(model)
      model = LeakyReLU(alpha=0.2)(model)
      model = Dropout(0.5)(model)
      model = Dense(1024)(model)
      model = LeakyReLU(alpha=0.2)(model)
      model = Dropout(0.5)(model)
      validity = Dense(1, activation="sigmoid")(model)

      return Model([z, img], validity)


def train():
  img_shape = (28, 28, 1)
  latent_dim = 100

  optimizer = Adam(0.0002, 0.5)

  # Build and compile the discriminator
  GAN =biGAN()

  D = GAN.discriminator()
  D.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

  G = GAN.generator()

  # Build the encoder
  encoder = GAN.encoder()

  # The part of the bigan that trains the discriminator and encoder
  D.trainable = False

  # Generate image from sampled noise
  z = Input(shape=(latent_dim, ))
  img_ = G(z)

  # Encode image
  img = Input(shape=img_shape)
  z_ = encoder(img)

  # Latent -> img is fake, and img -> latent is valid
  fake = D([z, img_])
  valid = D([z_, img])

  # Set up and compile the combined model
  # Trains generator to fool the discriminator
  bigan_generator = Model([z, img], [fake, valid])
  bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
    optimizer=optimizer)


  D.summary()
  encoder.summary()
  bigan_generator.summary()

  epochs=40000
  batch_size=32
  sample_interval=20

  # Load the dataset
  (X_train, _), (_, _) = mnist.load_data()

  # Rescale -1 to 1
  X_train = (X_train.astype(np.float32) - 127.5) / 127.5
  X_train = np.expand_dims(X_train, axis=3)

  # Adversarial ground truths
  valid = np.ones((batch_size, 1))
  fake = np.zeros((batch_size, 1))

  for epoch in range(epochs):

      # Sample noise and generate img
      z = np.random.normal(size=(batch_size, latent_dim))
      imgs_ = G.predict(z)

      # Select a random batch of images and encode
      idx = np.random.randint(0, X_train.shape[0], batch_size)
      imgs = X_train[idx]
      z_ = encoder.predict(imgs)

      # Train the discriminator (img -> z is valid, z -> img is fake)
      d_loss_real = D.train_on_batch([z_, imgs], valid)
      d_loss_fake = D.train_on_batch([z, imgs_], fake)
      d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

      # ---------------------
      #  Train Generator
      # ---------------------

      # Train the generator (z -> img is valid and img -> z is is invalid)
      g_loss = bigan_generator.train_on_batch([z, imgs], [valid, fake])

      # Plot the progress
      print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

      # If at save interval => save generated image samples
      if epoch % sample_interval == 0:
          gene = G.predict(z)
          Ds = image.array_to_img((gene[0] * 127.5)+127.5, scale=False)
          plt.imshow(Ds)
          plt.show()

if __name__ == '__main__':
    train()