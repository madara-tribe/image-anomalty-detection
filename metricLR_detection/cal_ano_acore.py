import keras
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import cv2
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from model import metric_model
from train import *

normal_path='normal_class'
ano_path = 'anomaly_class'
x_normal, y_normal, x_test_normal, y_test_normal= create_data(normal_path,
                                                        label_number = 0, split = 800)

x_ano, y_ano, x_test_ano, y_test_ano = create_data(ano_path,
                                                        label_number = 0, split = 800)

base_model = metric_model()
base_model.load_weights("weight/metariclr.h5")
print("metric lerning model")
base_model.layers.pop() # delete last layer
metricLR_model = Model(inputs=base_model.input,outputs=base_model.layers[-1].output)
metricLR_model.summary()


def anomaly_dataframe_histgram():
    print("total ano score data histgram and plot") 
    def create_pred_data(model, images):
      y_pred = [model.predict(im.reshape(1, 256, 256, 3)) for im in images]
      y_pred =[a.max() for a in y_pred]
      return np.array(y_pred)


    def plot_anomaly():
      train_df = pd.DataFrame(normal_pred, columns=['train'])
      test_ndf = pd.DataFrame(test_normal_pred, columns=['normal'])
      test_anodf = pd.DataFrame(ano_pred, columns=['anomaly'])
      print(train_df.shape, test_anodf.shape, test_ndf.shape)
      sns.distplot(train_df["train"], kde=True, rug=True, color = 'blue', label = 'train_normal')
      sns.distplot(test_anodf["anomaly"], kde=True, rug=True, color = 'red', label = 'test_anomaly')
      sns.distplot(test_ndf["normal"], kde=True, rug=True, color = 'green', label = 'test_normal')
      plt.legend()
      plt.show()

    normal_pred = create_pred_data(metricLR_model, x_normal)
    test_normal_pred = create_pred_data(metricLR_model, x_test_normal)
    ano_pred = create_pred_data(metricLR_model, x_ano)
    plot_anomaly()





def plot_ano_score():
  print("calculate and plot ano score for each data")
  dim = x_normal.shape[1]
  ano=[]
  for a in x_ano:
    a= a*1000
    ano_score = -clf._decision_function(a.reshape(1, dim))
    ano.append(ano_score)
  
  normal=[]
  for b in x_test_normal:
    b= b*1000
    normal_score = -clf._decision_function(b.reshape(1, dim))
    normal.append(normal_score)


  ano_df = pd.DataFrame(ano, columns=['anomaly'])
  ndf = pd.DataFrame(normal, columns=['normal'])
  print(ano_df.shape, ndf.shape)

  sns.distplot(ano_df["anomaly"], kde=True, rug=True, color = 'red', label = 'anomaly')
  sns.distplot(ndf["normal"], kde=True, rug=True, color = 'green', label = 'normal')
  plt.legend()
  plt.show()



def MinMaxScaler_ano_score():
  print("calculate ano score between normal and anomaly")

  def create_score_data(model, images):
    pred_images = model.predict(images) # train images
    score_data = pred_images.reshape((len(pred_images),-1))
    return score_data

  train_normal = create_score_data(metricLR_model, x_normal)
  test_normal = create_score_data(metricLR_model, x_test_normal) # test anomaly 
  test_ano = create_score_data(metricLR_model, x_ano) # test normal
  print(train_a.shape, test_a.shape, test_b.shape)

  # MinMaxScaler

  ms = MinMaxScaler()
  train_normal = ms.fit_transform(train_normal)
  clf = LocalOutlierFactor(n_neighbors=5)
  clf.fit(train_normal)


  test_normal = ms.transform(test_normal)
  test_ano = ms.transform(test_ano)
  Z1 = -clf._decision_function(test_normal)
  Z2 = -clf._decision_function(test_ano)
  print('ano score {}, normal score {}'.format(sum(Z1), sum(Z2)))


def LOF_ano_score():
  print("each ano score by LOF predict method range -1 to +1")
  lof = LocalOutlierFactor(n_neighbors=10,
                            novelty=True, contamination=0.1)

  lof.fit(train_normal)
  # each LOF prediction label (-1 is anomaly and 1 is normal)
  test_a_pred = lof.predict(test_normal) # テストデータに対する予測
  test_b_pred = lof.predict(test_ano) 
  print(test_a_pred, test_b_pred)



def MCD_ano_score():
  print("マハラノビス距離(each MCD) ano score")
  mcd = MinCovDet()
  mcd.fit(train_normal)
  mcd_anoscore = mcd.mahalanobis(test_normal)
  mcd_normalscore = mcd.mahalanobis(test_ano)
  print("mcd ano score {} mcd normal score {}".format(mcd_anoscore, mcd_normalscore))




if __name__ == '__main__':
  """cal anomaky score by various ways"""
  anomaly_dataframe_histgram()
  plot_ano_score()
  MinMaxScaler_ano_score()
  LOF_ano_score()
  MCD_ano_score()