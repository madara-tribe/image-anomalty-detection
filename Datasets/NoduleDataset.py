from google.colab import drive
drive.mount('/content/drive')

!pip install pydicom pypng
import png
import pydicom
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

# save

dir_path= 'drive/My Drive/NonNodule93images'
output_folder= 'drive/My Drive/Ds'
D=[]
for i, doc in enumerate(os.listdir(dir_path)):
  ds = pydicom.dcmread(os.path.join(dir_path, doc))
  shape = ds.pixel_array.shape

  # Convert to float to avoid overflow or underflow losses.
  image_2d = ds.pixel_array.astype(float)

  # Rescaling grey scale between 0-255
  image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

  # Convert to uint
  image_2d_scaled = np.uint8(image_2d_scaled)
  with open(os.path.join(output_folder, doc)+'{}_.png'.format(i) , 'wb') as png_file:
    w = png.Writer(shape[1], shape[0], greyscale=True)
    w.write(png_file, image_2d_scaled)

    
# 2th way
dir_path= 'drive/My Drive/NonNodule93images'
def trainGenerator(img_path):
  img=[]
  for im in os.listdir(img_path):
    im = pydicom.read_file(img_path+'/'+im)
    imgs = im.pixel_array
    if imgs is not None:
      # imgs = cv2.resize(imgs, (256, 256))
      img.append(imgs)
  inputs=np.array(img)
  return inputs


x_train=trainGenerator(dir_path)

save_dir = 'drive/My Drive/Ds'
for i, im in enumerate(x_train):
    img = image.array_to_img(im.reshape(2048,2048,1), scale=True)
    img.save(os.path.join(save_dir, 'Nodule_{}.png'.format(i)))
