from os import listdir

import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras import Model, models
from matplotlib import pyplot
#trained = models.load_model('model_055200.h5')
def load_images(path_dir, size=(256, 512)):
    src_list, tar_list = list(), list()
    for filename in listdir(path_dir):
        if '.jpg' in filename:
            pixels = load_img(path_dir + filename, target_size=size)
            pixels = img_to_array(pixels)
            src_list.append(pixels[:, :256])
            tar_list.append(pixels[:, 256:])
    return [np.asarray(src_list), np.asarray(tar_list)]

def load_images_test(path_dir, size=(256, 256)):
    src_list = list()
    for filename in listdir(path_dir):
        if '.jpg' in filename:
            pixels = load_img(path_dir + filename, target_size=size)
            pixels = img_to_array(pixels)
            src_list.append(pixels[:, :256])
    return [np.asarray(src_list)]
path = "archive/facades/facades/test/"


[src_images, tar_images] = load_images(path)
filename = 'facades_test_256.npz'
np.savez_compressed(filename, src_images)
print('Saved dataset: ', filename)

#print(src_images.shape)

def load_image (filename, size=(256, 512)):
    pixels = load_img(filename, target_size=size)
    pixels = img_to_array(pixels)
    pixels = pixels[:, :256]
    pixels = (pixels - 127.5)/127.5
    pixels = np.expand_dims(pixels, 0)
    return pixels

def load_real_samples(filename):
    data = np.load(filename)
    print(data)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]



[src, tar] = load_real_samples('facades_test_256.npz')
trained = models.load_model('model_156000.h5', compile=False)
gen = trained.predict(src)
src = (src + 1) / 2.0
gen = (gen + 1) / 2.0
tar = (tar + 1) / 2.0
# plot the image
pyplot.subplot(3, 1, 1)
pyplot.imshow(gen[0])
pyplot.subplot(3, 1, 2)
pyplot.imshow(src[0])
pyplot.subplot(3, 1, 3)
pyplot.imshow(tar[0])

pyplot.axis('off')
pyplot.show() pow()

