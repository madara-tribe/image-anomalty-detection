from keras.utils import np_utils
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras

class VAE:
    def vae_net(self):
        latent_dim=2
        inputs = Input(shape=(256, 256, 1), name='encoder_input')
        x = Conv2D(64, kernel_size=2, strides=2)(inputs)
        x = Conv2D(64, kernel_size=2, strides=2)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(128, kernel_size=2, strides=2)(inputs)
        x = Conv2D(128, kernel_size=2, strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256, kernel_size=2, strides=2)(x)
        x = Conv2D(256, kernel_size=2, strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)

        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(4*4)(latent_inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((4,4,1))(x)

        x = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(x)
        x = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(x)
        x = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(x)
        x = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x1 = Conv2DTranspose(1, kernel_size=4, padding='same')(x)
        x1 = BatchNormalization()(x1)
        out1 = Activation('sigmoid')(x1)#out.shape=(n,8,8,1)

        x2 = Conv2DTranspose(1, kernel_size=4, padding='same')(x)
        x2 = BatchNormalization()(x2)
        out2 = Activation('sigmoid')(x2)#out.shape=(n,8,8,1)

        decoder = Model(latent_inputs, [out1, out2], name='decoder')

        # build VAE model
        outputs_mu, outputs_sigma_2 = decoder(encoder(inputs)[2])
        vae = Model(inputs, [outputs_mu, outputs_sigma_2], name='vae_mlp')
        # VAE_loss
        vae_loss=self.vae_loss(inputs, outputs_mu, outputs_sigma_2, z_mean, z_log_var)
        return vae, vae_loss

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    def vae_loss(self, inputs, outputs_mu, outputs_sigma_2, z_mean, z_log_var):
        # VAE loss
        m_vae_loss = (K.flatten(inputs) - K.flatten(outputs_mu))**2 / K.flatten(outputs_sigma_2)
        m_vae_loss = 0.5 * K.sum(m_vae_loss)

        a_vae_loss = K.log(2 * 3.14 * K.flatten(outputs_sigma_2))
        a_vae_loss = 0.5 * K.sum(a_vae_loss)

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        vae_loss = K.mean(kl_loss + m_vae_loss + a_vae_loss)
        return vae_loss
