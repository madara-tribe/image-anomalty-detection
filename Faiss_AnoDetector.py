import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from styleGAN.styleGAN import styleGAN
import faiss
import collections
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
from train import load_dataset


normal_path= 'normal_class'
ano_path = 'anomaly_class'


def plot_confusion_matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def return_pred_label(I, dataset_label, candidate):
    y_pred = []
    one=1
    zero=0
    for i in I:
        c = collections.Counter(dataset_label[i])
        if c[one]> candidate/2:
            y_pred.append(one)
        else:
            y_pred.append(zero)
    return y_pred

  
def feature_extractor():
    alpha = 5
    path ="styleGAN/tb"
    GAN = styleGAN()
    D = GAN.discriminator()
    D.load_weights(os.path.join(path, 'discriminator.h5'))
    return D


def create_data(x_train, label_number, split = 800):

    x_train = x_train[:split]
    x_valid = x_train[split:]
  
    k = np.array([label_number]*len(x_train))
    y_train = np.reshape(k, (len(x_train),))

    y_train = y_train[:split]
    y_valid = y_train[split:]

    return x_train, y_train, x_valid, y_valid


def faiss_anomaly_detection(candidate=50):
    dim = normal_feature.shape[1]
    index = faiss.IndexFlatL2(dim)   
    index.add(normal_feature)
    # search
    D, I = index.search(ano_feature, candidate)
    # evaluate
    y_pred = return_pred_label(I, y_normal, candidate)

    """confusion_matrix and accuracy"""
    cm = confusion_matrix(ano_aquery_label, y_pred)
    plot_confusion_matrix(cm, classes='anomaly',
                        title='Confusion matrix, without normalization')
    # accuracy
    print('acuracy:{}'.format(accuracy_score(ano_aquery_label, y_pred)))
    #label_string = ['{}'.format(i) for i in range(2)]
    print(classification_report(ano_aquery_label, y_pred))


if __name__ == '__main__':
    print('load normal query images')
    x_train = load_dataset(normal_path)
    x_normal, x_query, y_normal, y_query = create_data(x_train, label_number=0, split = 900)

    print('load anomaly query images')
    x_test = load_dataset(ano_path)
    x_ano, ano_query, y_ano, ano_aquery_label = create_data(x_test, label_number=1, split = 100)
    print(anonormal_dataset.shape, query_anomaly.shape)


    """Pridict"""
    print('predict normal dataset')
    model = feature_extractor()
    normal_feature = model.predict(x_normal)
    print(normal_feature.shape)


    print('predict anomaly query')
    ano_feature = model.predict(ano_query)
    print(ano_feature.shape) 

    # faiss anomaly detection
    faiss_anomaly_detection(candidate=50)