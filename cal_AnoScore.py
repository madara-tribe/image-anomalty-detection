import os
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from styleGAN.styleGAN import styleGAN
from tools.LOF_Score import LOF_Score, lof_each_ano_score
from tools.MCD_Score import MCD_Score
from train import load_dataset


GAN = styleGAN()
D = GAN.discriminator()
D.load_weights("styleGAN/tb/discriminator.h5")


def predict_image(images):
    y_pred=[]
    for im in images:
        array = D.predict(im.reshape(1, 256, 256, 3))
        y_pred.append(float(array))
    return np.array(y_pred)


def plot_anomaly_histgram(train_normal, test_normal, anomaly_test):
    normal_pred = predict_image(train_normal)
    test_normal_pred = predict_image(test_normal)
    ano_pred = predict_image(anomaly_test)


    print("make query dataframe and plot")

    train_ndf = pd.DataFrame(normal_pred, columns=['train'])
    test_ndf = pd.DataFrame(test_normal_pred, columns=['normal'])
    test_anodf = pd.DataFrame(ano_pred, columns=['anomaly'])
    print(train_ndf.shape, test_anodf.shape, test_ndf.shape)

    # plot 
    sns.distplot(train_ndf["train"], kde=True, rug=True, color = 'blue', label = 'train_normal')
    sns.distplot(test_ndf["normal"], kde=True, rug=True, color = 'green', label = 'test_normal')
    sns.distplot(test_anodf["anomaly"], kde=True, rug=True, color = 'red', label = 'test_anomaly')
    plt.legend()
    plt.show()



def cal_anoo_score(train_normal, anomaly_test, test_normal):
    train_a, test_a, test_b = LOF_Score(train_normal, anomaly_test, test_normal)
    print("each ano score by LOF predict method range -1 to +1")
    lof_each_ano_score(train_a, test_a, test_b)

    print("each MCD(マハラノビス距離) ano score")
    MCD_Score(train_a, test_a, test_b)


if __name__ == '__main__':
    normal_path= 'normal_class'
    ano_path = 'anomaly_class'

    print('load normal query images')
    x_train = load_dataset(normal_path)
    train_normal = x_train[:900]
    test_normal = x_train[900:]

    print('load anomaly query images')
    anomaly_test = load_dataset(ano_path)

    # plot histgram 
    plot_anomaly_histgram(train_normal, test_normal, anomaly_test)
    # calculate anomaly score
    cal_anoo_score(train_normal, anomaly_test, test_normal)