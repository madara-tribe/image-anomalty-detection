from __future__ import print_function, division
import time
from keras import Input
from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from keras.preprocessing import image


latent_dim=100

class GANs:
    def discriminator(self):
    
        discriminator_input = Input(shape=(256,256,3))
        x = Conv2D(128, 3)(discriminator_input)
        x = LeakyReLU()(x)
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)

        # One dropout layer - important trick!
        x = Dropout(0.4)(x)

        # Classification layer
        out = Dense(1, activation='sigmoid')(x)

        return Model(discriminator_input, out)

    def generator(self):
        latent_dim=100
        generator_input = Input(shape=(latent_dim,))

        # First, transform the input into a 16x16 128-channels feature map
        x = Dense(128 * 128 * 128)(generator_input)
        x = LeakyReLU()(x)
        x = Reshape((128, 128, 128))(x)

        # Then, add a convolution layer
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        # Upsample to 32x32
        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

        # Few more conv layers
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        # Produce a 32x32 1-channel feature map
        output = Conv2D(3, 7, activation='tanh', padding='same')(x)
        return Model(generator_input, output)




def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # discriminator

    GAN=GANs()
    discriminator=GAN.discriminator()
    generator=GAN.generator()
    discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')
    discriminator.summary()
    generator.summary()

    # gan

    discriminator.trainable = False # adjust to only gan

    gan_input=Input(shape=(latent_dim,))
    gan_output=discriminator(generator(gan_input))
    gan=Model(gan_input, gan_output)
    gan.compile(optimizer=Adam(lr=0.0001, beta_1=0.5), loss='binary_crossentropy')
    gan.summary()


    class trainGenerator(object):
        def __init__(self):
            self.img_generator=[]
        
        def flow_from_dir(self, img_path):
        for imgs in os.listdir(img_path):
                imgs=cv2.imread(img_path+'/'+imgs)

                if imgs is not None:
                    imgs = cv2.resize(imgs, (256, 256))
                    self.img_generator.append(imgs/ 255)
        input_img=np.array(self.img_generator, dtype=np.float32)
        return input_img


    train=trainGenerator()
    x_train=train.flow_from_dir('/home/ec2-user/gan_train_jpg')
    print(x_train.shape)



    iterations = 10000
    batch_size = 90
    counter=1
    save_dir = '/home/ec2-user/logss/'

    # Start training loop
    for step in range(iterations):
    batch_idx= min(len(x_train), np.inf) // batch_size
    for idx in range (0, batch_idx):
        real_images = x_train[idx*batch_size:(idx+1)*batch_size]
        # Sample random points in the latent space
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        # Decode them to fake images
        generated_images = generator.predict(random_latent_vectors)

        # Combine them with real images
        combined_images = np.concatenate([generated_images, real_images])

        # Assemble labels discriminating real from fake images
        labels = np.concatenate([np.ones((batch_size, 1)),
                                np.zeros((batch_size, 1))])
        # Add random noise to the labels - important trick!
        labels += 0.05 * np.random.random(labels.shape)

        # Train the discriminator
        d_loss = discriminator.train_on_batch(combined_images, labels)

        # sample random points in the latent space
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        # Assemble labels that say "all real images"
        misleading_targets = np.zeros((batch_size, 1))

        # Train the generator (via the gan model,
        # where the discriminator weights are frozen)
        g_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
        print("Epoch: [%2d] [%4d/%4d], g_loss: %s, d_loss: %s" % (step, idx, batch_size, g_loss, d_loss))
        counter += 1
        
        if np.mod(counter, 500)==1:
            # Save model weights
            generator.save_weights('/home/ec2-user/tb/generator.h5')
            discriminator.save_weights('/home/ec2-user/tb/discriminator.h5')
            # Print metrics
            print('discriminator loss at step %s: %s' % (step, d_loss))
            print('adversarial loss at step %s: %s' % (step, g_loss))

            # Save one generated image
            img = image.array_to_img(generated_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))

            # Save one real image, for comparison
            img = image.array_to_img(real_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))

if __name__ == '__main__':
    train()