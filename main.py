import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input, Concatenate, Conv2D, LeakyReLU, BatchNormalization, Activation, Conv2DTranspose, Dropout
from keras import Model
from keras.optimizers import Adam
from os import listdir

def load_images(path, size = (256, 512)):
    src_list, tar_list = list(), list()
    for filename in listdir(path):
        if '.jpg' in filename:
            pixels = load_img(path + filename, target_size= size)
            pixels = img_to_array(pixels)
            src_list.append(pixels[:, :256])
            tar_list.append(pixels[:, 256:])
    return [np.asarray(src_list), np.asarray(tar_list)]


path = "archive/maps/maps/train/"

[src_images, tar_images] = load_images(path)

print(src_images.shape, tar_images.shape)

def discriminator(shape=(256, 256, 3)):
    in_img = Input(shape=shape)
    tar_img = Input(shape=shape)
    merged_img = Concatenate()([in_img, tar_img])
    layer = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(merged_img)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alph=0.2)(layer)

    layer = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(512, (4, 4), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(1, (4, 4), padding='same')(layer)
    result = Activation('sigmoid')(layer)

    model = Model([in_img, tar_img], result)
    opt = Adam(lr=0.002, beta=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


def encoder(layer, no_filter, batch=True):
    layer_encode = Conv2D(no_filter, (4, 4), strides=(2, 2), padding='same')(layer)
    if batch:
        layer_encode = BatchNormalization()(layer_encode)
    layer_encode = LeakyReLU(alpha=0.2)(layer_encode)
    return layer_encode

def decoder(layer, skip, no_filter, dropout=True):
    layer_decode = Conv2DTranspose(no_filter, (4, 4), strides=(2, 2), padding='same')(layer)
    layer_decode = BatchNormalization()(layer_decode)
    if dropout:
        layer_decode = Dropout(0.5)(layer_decode)
    layer_decode = Concatenate()([layer_decode, skip])
    layer_decode = Activation('relu')(layer_decode)
    return layer_decode

def generator():


