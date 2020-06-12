from keras.callbacks import *
import keras
import matplotlib.pyplot as plt
import keras.backend as K
import keras
import math
import os
import numpy as np
import cv2
from model import metric_model



def load_data(img_path):
  img=[]
  for idx, imgs in enumerate(os.listdir(img_path)):
    imgs=cv2.imread(img_path+'/'+imgs)
    if imgs is not None:
      img.append(imgs/255)
  inputs=np.array(img)
  return inputs


class Generator(object):
    def __init__(self):
        self.img_generator=[]
        self.label_generator=[]
        self.reset()
   
    def reset(self):
        self.img_generator=[]
        self.label_generator=[]
      
    def flow_from_dir(self, imgs, labels, batch_size=10):
        while True:
            for img, label in zip(imgs, labels):
                if img is not None:
                  self.img_generator.append(img)
                if label is not None:
                  self.label_generator.append(label)
                    
                  if len(self.img_generator)==batch_size:
                      input_img=np.array(self.img_generator)
                  if len(self.label_generator)==batch_size:
                      input_label=np.array(self.label_generator)
                      self.reset()
                      yield input_img, input_label
 



def create_data(folder_path, label_number, split = 800):
    # train(normal)
    x_train = load_data(folder_path)
    print(x_train.max(), x_train.min())

    x_train = x_train[:split]
    x_valid = x_train[split:]
  
    k = np.array([label_number]*len(x_train))
    y_train = np.reshape(k, (len(x_train),))

    y_train = y_train[:split]
    y_valid = y_train[split:]

    return x_train, y_train, x_valid, y_valid





def train():
    normal_path='normal_class'
    x_train, y_train, x_valid, y_valid = create_data('normal_class', label_number=0)

    # generator
    train = Generator()
    val = Generator()
    training=train.flow_from_dir(x_train, y_train)
    validations=val.flow_from_dir(x_valid, y_valid)

    # model
    metriclr_models = metric_model()
    metriclr_models.summary()
    
    def step_decay(epoch):
      initial_lrate = 0.01
      decay_rate = 0.5
      decay_steps = 8.0
      lrate = initial_lrate * math.pow(decay_rate,  
             math.floor((1+epoch)/decay_steps))
      return lrate

    # callback
    callback=[]
    callback.append(HistoryCheckpoint(filepath='tb/LearningCurve_{history}.png', verbose=1, period=2))
    callback.append(LearningRateScheduler(step_decay))
    callback.append(TensorBoard(log_dir='tb/'))
    callback.append(ModelCheckpoint('logss/{epoch:02d}_metric.hdf5', monitor='loss', verbose=1))

    # train
    try:
      metriclr_models.fit_generator(training, steps_per_epoch=5000, epochs=2,
                        callbacks=callback, validation_data=validations, validation_steps=100)
    finally:
      metriclr_model.ssave('weight/metariclr.h5')


if __name__ == '__main__':
    train()