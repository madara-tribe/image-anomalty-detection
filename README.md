# Image_anomalty_detection

# Steps

## 1. Train by styleGAN

Train only normal images by stleGAN and generate too similar normal images

<b>Generating process</b>

![60765275-07bed000-a0d3-11e9-8096-63fa08a4c36c](https://user-images.githubusercontent.com/48679574/84467695-dc1b7100-acb7-11ea-9cc0-a0d1541671c7.GIF)


<b>Generated images</b>

![60763975-b30f5b00-a0ba-11e9-8f23-ca78c12721b1](https://user-images.githubusercontent.com/48679574/84467703-e0e02500-acb7-11ea-858e-da572a164309.png)


## 2.Calculate anomaly score and plot these by using trained model

With trained model, calculate anomaly score and plot these by various ways.

<b>Normal and anomaly image</b>

<img src="https://user-images.githubusercontent.com/48679574/84467756-053c0180-acb8-11ea-9adf-7cd0b547d7c3.jpg" width=50%>

<b>Plot these images as anomaly score</b>

<img src="https://user-images.githubusercontent.com/48679574/84467758-0705c500-acb8-11ea-9f62-c9323e33a31c.png" width=50%>


## 3.Anomaly detection with similar image search tec(faiss)

I acomplished about 99% accuracy to classify normal and anomaly images by stylaGAN model and similar image search tec(faiss)


<img src="https://user-images.githubusercontent.com/48679574/84467759-09681f00-acb8-11ea-92f5-0e6b0babfc95.png" width=50%>




# Summary
The logics and process are written by bellow my blog

https://trafalbad.hatenadiary.jp/entry/2019/09/04/213627
