from builtins import range
from os import listdir
from PIL import Image
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras import Model, models
from matplotlib import pyplot
import cv2
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import random
# trained = models.load_model('model_055200.h5')
def load_images(path_dir, size=(256, 512)):
    src_list, tar_list = list(), list()
    for filename in listdir(path_dir):
        if '.PNG' in filename:
            pixels = load_img(path_dir + filename, target_size=size)
            pixels = img_to_array(pixels)
            src_list.append(pixels[:, :256])
            tar_list.append(pixels[:, 256:])
    return [np.asarray(src_list), np.asarray(tar_list)]




def load_real_samples(filename):
    data = np.load(filename)
    print(data)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]




def convert_to_bw_and_form_npz(path_dir, size=(256, 256)):
    count = 0
    for file in listdir(path_dir):
        if '.JPG' in file:
            img = cv2.imread(path_dir + file)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 256))
            cv2.imwrite("test/Jan7/monochrome/" + str(count) + ".PNG", image)
            count += 1





def concatenate(original, annotated, concatenated, ran):
    for i in range(650, 1000):
        print(i)
        original_im = cv2.imread(original + 'frame_' + str(i).zfill(6) + '.PNG')
        original_im = cv2.resize(original_im, (256, 256))
        annotated_im = cv2.imread(annotated + 'mask_' + str(i).zfill(6) + '.PNG')
        annotated_im = cv2.resize(annotated_im, (256, 256))
        concat = cv2.hconcat([original_im, annotated_im])
        cv2.imwrite(concatenated + 'image_' + str(i).zfill(6) + '.PNG', concat)
    print('done')


def create_npz(path, filename):
    print('started')
    [src_images, tar_images] = load_images(path)
    np.savez_compressed(filename, src_images, tar_images)
    print('Saved dataset: ', filename)
    print(src_images.shape, tar_images.shape)




def test():
    [src, tar] = load_real_samples('thoracic_binary_test.npz')
    print(src.shape)
    print(tar.shape)

    trained = models.load_model('thoracic/trained/trained_model.h5', compile=False)
    gen = trained.predict(src)
    gen = (gen + 1) / 2.0
    tar = (tar + 1) / 2.0
    for i in range(len(gen)):
        concat = cv2.hconcat([gen[i], tar[i]])
        pyplot.imsave('thoracic/test/' + str(i) + '.PNG', concat)
    print(IOU(gen, tar))

def IOU(gen, tar):


    dice = np.empty(gen.shape[0])
    gen *= 255
    tar *= 255
    gen = np.uint8(gen)
    tar = np.uint8(tar)
    for i in range(len(gen)):
        threshgen = cv2.cvtColor(gen[i], cv2.COLOR_BGR2GRAY)
        threshtar = cv2.cvtColor(tar[i], cv2.COLOR_BGR2GRAY)
        print(np.unique(threshtar))
        arr = [0, 128, 257] #mean values based on np.unique(tar[i])
        prev = arr[0]
        cumulative = 0

        for val in arr[1:]:
            total = 0
            total += len(threshgen[(threshgen < val) & (threshgen >= prev)])
            total += len(threshtar[(threshtar < val) & (threshtar >= prev)])
            intersection = len(threshgen[(threshtar < val) & (threshtar >= prev) & (threshgen < val)
                                                                                         & (threshgen >= prev)])
            total = total - intersection
            cumulative += intersection/total

            #print(intersection/total) #individiual class scores

            prev = val


        dice[i] = cumulative*100/(len(arr) - 1)
        print(dice[i])
    return np.nanmean(dice)


def split(vid1, vid2):
    train1 = list()
    train2 = list()
    for filename in listdir(vid1):
        pixels = load_img(vid1 + filename)
        pixels = img_to_array(pixels)
        train1.append(pixels)

    random.shuffle(train1)

    for filename in listdir(vid2):
        pixels = load_img(vid2 + filename)
        pixels = img_to_array(pixels)
        train2.append(pixels)

    random.shuffle(train2)
    x = int(len(train1)*0.8)
    y = int(len(train2)*0.8)
    src_list, tar_list = list(), list()
    for i in range(x):
        src_list.append(train1[i][:, :256])
        tar_list.append(train1[i][:, 256:])
    for i in range(y):
        src_list.append(train2[i][:, :256])
        tar_list.append(train2[i][:, 256:])

    src_list = np.asarray(src_list)
    tar_list = np.asarray(tar_list)
    np.savez_compressed('3class_train.npz', src_list, tar_list)
    print(src_list.shape)


    src_list, tar_list = list(), list()
    for i in range(x, len(train1)):
        src_list.append(train1[i][:, :256])
        tar_list.append(train1[i][:, 256:])
    for i in range(y, len(train2)):
        src_list.append(train2[i][:, :256])
        tar_list.append(train2[i][:, 256:])

    src_list = np.asarray(src_list)
    tar_list = np.asarray(tar_list)
    np.savez_compressed('3class_test.npz', src_list, tar_list)
    print(tar_list.shape)


test()