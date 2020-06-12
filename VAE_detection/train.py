# https://qiita.com/shinmura0/items/811d01384e20bfd1e035

import os
import math
import scipy.misc
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from VAE import VAE
from toools.HistoryCallbackLoss import HistoryCheckpoint
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

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

def vae_train():   
    # load train
    x_train = load_dataset('adi')

    # load test normal and anomaly
    test_anomaly = load_dataset('vans')
    test_normal = load_dataset('adi_test')

    # drfine train and valid iamge for train
    trains = x_train[10:]
    valid = x_train[:10]
    print(trains.shape, valid.shape, test_anomaly.shape, test_normal.shape)

    # try to plot
    plt.imshow(x_train[10].reshape(256, 256))
    plt.gray()
    plt.show()


    # train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    def step_decay(epoch):
        initial_lrate = 0.0001
        decay_rate = 0.5
        decay_steps = 8.0
        lrate = initial_lrate * math.pow(decay_rate,  
            math.floor((1+epoch)/decay_steps))
        return lrate


    callback=[]
    callback.append(HistoryCheckpoint(filepath='tb/LearningCurve_{history}.png', verbose=1, period=300))
    callback.append(LearningRateScheduler(step_decay))


    model = VAE()
    model, loss=model.vae_net()
    #model.load_weights("vae_model.h5")

    model.add_loss(loss)
    model.compile(optimizer =Adam(lr=0.0001))
    model.summary()

    try:
        model.fit(trains, batch_size=20, epochs=300, 
            callbacks=callback, validation_data=(valid, None))
    finally:
        model.save('weight/vae_model.h5')


if __name__ == '__main__':
    train()