from styleGAN import styleGAN
from train import noise, noiseImage
import numpy as np
import math



# Style Z
def noise(batch_size):
    return np.random.normal(0.0, 1.0, size = [batch_size, latent_size])

# Noise Sample
def noiseImage(batch_size):
    return np.random.uniform(0.0, 1.0, size = [batch_size, im_size, im_size, 1])


def generate(batch_size):
    GAN = styleGAN()
    G = GAN.generator()
    G.load_weights('tb/generator.h5')
    n = noise(batch_size)
    n2 = noiseImage(batch_size)
    generated_images = G.predict([n, n2, np.ones([batch_size, 1])])
    generated_images = np.array(generated_images, dtype=np.float32)
    return generated_images

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                    dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
    return image
  

def evaluate(): 
    num_images = 25
    # generate images
    generated_imgs = generate(num_images)
    combined_imgs = combine_images(generated_imgs)
    # plot 
    plt.imshow(combined_imgs)
    plt.show()


if __name__ == '__main__':
    evaluate()