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
from styleGAN.loss import gradient_penalty_loss
from styleGAN.AdaIN import AdaInstanceNormalization
from styleGAN.styleGAN import styleGAN
from train import load_dataset
IMAGE_SHAPE = 256

#Style Z
def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size])

#Noise Sample
def noiseImage(n):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1])


def create_detector_model():
    def feature_extractor():
        GAN = styleGAN()
        D = GAN.discriminator()
        D.load_weights('styleGAN/tb/discriminator.h5')
        intermidiate_model = Model(inputs=D.layers[0].input, outputs=D.layers[-6].output)
        intermidiate_model.compile(loss='mse', 
                                optimizer=Adam(lr=0.0002, beta_1 = 0, beta_2 = 0.99, decay = 0.00001))
        return intermidiate_model
  
    def sum_of_residual(y_true, y_pred):
        return K.sum(K.abs(y_true - y_pred))

    GAN = styleGAN()
    G = GAN.generator()
    G.load_weights('styleGAN/tb/generator.h5')
    intermidiate_model = feature_extractor()
    intermidiate_model.trainable = False
        
    # compute D(G(z))
    gi = Input(shape = [latent_size])
    gi2 = Input(shape = [im_size, im_size, 1])
    gi3 = Input(shape = [1])
    g_out = G([gi, gi2, gi3])
    d_out = intermidiate_model(g_out)
    model = Model(inputs=[gi, gi2, gi3], outputs=[g_out, d_out])
    model.compile(loss=sum_of_residual, loss_weights= [0.90, 0.10], 
                  optimizer=Adam(lr=0.0002, beta_1 = 0, beta_2 = 0.99, decay = 0.00001))
    K.set_learning_phase(0)
    
    return model
  
# from tb_his import HistoryCheckpoint
# callback=[]
# callback.append(HistoryCheckpoint(filepath='tb/LearningCurve_{history}.png', verbose=1, period = iterations))
def compute_anomaly_score(model, x, batch_size=1, iterations=500):
    n1 = noise(batch_size)
    n2 = noiseImage(batch_size)
    n3 = np.ones((batch_size, 1))
    intermidiate_model = feature_extractor()
    d_x = intermidiate_model.predict(x)

    # learning for changing latent
    loss = model.fit([n1, n2, n3], [x, d_x], batch_size=1, epochs=iterations, verbose=0)
    similar_data, _ = model.predict([n1, n2, n3])
    
    loss = loss.history['loss'][-1]
    
    return loss, similar_data

def plot_anomaly(similar, query, ano_score=None):
    Ls= similar.reshape(IMAGE_SHAPE, IMAGE_SHAPE, 3) - query
    np_residual = (Ls*255).astype(np.uint8)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
    
    original_x = (similar.reshape(IMAGE_SHAPE,IMAGE_SHAPE,3)*127.5+127.5).astype(np.uint8)
    #original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    show = cv2.addWeighted(original_x, 0.3, residual_color, 0.7, 0.)
    
    plt.figure(1, figsize=(3, 3))
    plt.title('query anomaly image')
    plt.imshow(query.reshape(IMAGE_SHAPE, IMAGE_SHAPE, 3), cmap=plt.cm.gray)
    
    if ano_score:
        print("anomaly score : ", ano_score)
      
    plt.figure(2, figsize=(3, 3))
    plt.title('generated normal image')
    plt.imshow(similar.reshape(IMAGE_SHAPE, IMAGE_SHAPE, 3), cmap=plt.cm.gray)

    plt.figure(3, figsize=(3, 3))
    plt.title('anomaly detection')
    plt.imshow(show)
    plt.show()




def plot_ano_by_trainedmodel():
    """
    Caliculate images diffference by weight and subtraction
    """  
    # load normalized anomaly image
    anomaly_image = load_dataset('anomaly_images')
    detector_model = create_detector_model()
    ano_score, similar_img = compute_anomaly_score(detector_model, anomaly_image[1].reshape(1, IMAGE_SHAPE, IMAGE_SHAPE, 3))
    plot_anomaly(similar_img, anomaly_image[1], ano_score=ano_score)





RESIZE_SHAPE = 64

def plot_1d_image(similar_img, ano_test):
    
    # convert to grayscale
    similar_imgs = cv2.cvtColor(similar_img.reshape(IMAGE_SHAPE, IMAGE_SHAPE, 3), cv2.COLOR_BGR2GRAY)
    ano_tests = cv2.cvtColor(ano_test.reshape(IMAGE_SHAPE, IMAGE_SHAPE, 3), cv2.COLOR_BGR2GRAY)

    # resize
    similar_imgs = cv2.resize(similar_imgs, (RESIZE_SHAPE, RESIZE_SHAPE))
    ano_tests = cv2.resize(ano_tests, (RESIZE_SHAPE, RESIZE_SHAPE))

    # convert 1-d
    Oned_similar = similar_imgs.flatten()
    Oned_anomaly_test = ano_tests.flatten()

    # modify normalization
    Oned_similar= Oned_similar*255
    Oned_anomaly_test= Oned_anomaly_test*255

    # plot image difference
    sns.distplot(Oned_similar, kde=True, rug=True, color = 'blue', label = 'similar')
    sns.distplot(Oned_anomaly_test, kde=True, rug=True, color = 'red', label = 'anomaly')
    plt.legend()
    plt.show()
    
    return Oned_similar, Oned_anomaly_test

def plot_dim1_image_diff(oned_similar, oned_anomaly):
    diff_img=[]
    diff_area = []
    for a, b in zip(oned_similar, oned_anomaly):
        if abs(a-b)>=1 and abs(a-b)<=10:
            diff_img.append(0)
            diff_area.append(1)
        else:
            diff_img.append(255)
            diff_area.append(0)

    diff_imgs = np.array(diff_img).reshape(RESIZE_SHAPE, RESIZE_SHAPE)
    print('ano score {}'.format(sum(diff_area)))
    plt.imshow(diff_imgs)
    plt.show()

def plot_1dim_image_anomaly():
    """
    Plot images diffference by 1-d and subtraction image
    """
    # plot by image by 1d 
    dim1_similar, dim1_anomaly = plot_1d_image(similar_img, anomaly_image)
    # plot difference form 1d to image and caliculate anomaly score from subtraction area
    plot_dim1_image_diff(dim1_similar, dim1_anomaly)
    


if __name__ == '__main__':
    plot_ano_by_trainedmodel()
    plot_1dim_image_anomaly()