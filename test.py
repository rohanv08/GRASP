from os import listdir
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras import Model, models
from matplotlib import pyplot
import cv2
from scipy import ndimage
import random
# trained = models.load_model('model_055200.h5')
def load_images(path_dir, ran = 1000, size=(256, 512)):
    src_list, tar_list = list(), list()
    for i in range(ran):
        filename = 'image_' + str(i).zfill(6) + '.PNG'
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
    for i in range(0, ran):
        print(i)
        original_im = cv2.imread(original + 'frame_' + str(i).zfill(6) + '.PNG')
        original_im = cv2.resize(original_im, (256, 256))
        annotated_im = cv2.imread(annotated + 'frame_' + str(i).zfill(6) + '.PNG')
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
    [src, tar] = load_real_samples('thoracic_multi.npz')
    trained = models.load_model('trained/trained_model_surgery_multi.h5', compile=False)
    gen = trained.predict(src)
    gen = (gen + 1) / 2.0
    src = (src + 1) / 2.0
    tar = (tar + 1) / 2.0
    for i in range(len(gen)):
         concat = cv2.hconcat([src[i], gen[i]])
         pyplot.imsave('thoracic/test/' + str(i) + '.PNG', concat)
     #with open('scores.txt', 'a') as f:
      #  f.write(str((IOU(gen, tar))))
      # f.write('\n')

def test_rotated_blurred(path, testSize, type, tilt):
    [src, tar] = load_real_samples(path)
    index_shuf = list(range(len(src)))
    random.shuffle(index_shuf)
    src_new = list()
    tar_new = list()
    for i in range(testSize):
        src_new.append(ndimage.rotate(src[index_shuf[i]], tilt, reshape=False))
        #print(np.unique(ndimage.rotate(src[index_shuf[i]], 10, reshape=False)))
        tar_new.append(ndimage.rotate(tar[index_shuf[i]], tilt, reshape=False))
        #cv2.imshow("image", ndimage.rotate(src[index_shuf[i]], tilt, reshape=False))
        #cv2.waitKey(0)

        #tar_new.append(cv2.rotate(tar[index_shuf[i]], cv2.cv2.ROTATE_90_CLOCKWISE))
        #src_new.append(src[index_shuf[i]])
        #tar_new.append(tar[index_shuf[i]])

    src_new = np.asarray(src_new)
    tar_new= np.asarray(tar_new)
    print(src_new.shape)
    print(tar_new.shape)
    src_new[src_new > 1] = 1
    src_new[src_new < -1] = -1
    tar_new[tar_new < -1] = -1
    tar_new[tar_new > 1] = 1
    #print(np.unique(tar_new[0]))
    trained = models.load_model('trained/trained_model_multiclass.h5', compile=False)
    gen = trained.predict(src_new)
    gen = (gen + 1) / 2.0
    tar_new = (tar_new + 1) / 2.0
    #cv2.imshow("image", tar_new[0])
    #cv2.waitKey(0)
    #print(np.unique(tar_new))

    for i in range(len(gen)):
       concat = cv2.hconcat([gen[i], tar_new[i]])
       pyplot.imsave('testImages/' + str(i) + '.PNG', concat)

    with open('scores.txt', 'a') as f:
        f.write(str((IOU(tar_new, gen))) + type)
        f.write('\n')





def IOU(gen, tar):
    dice = np.empty(gen.shape[0])
    gen *= 255
    tar *= 255
    gen = np.uint8(gen)
    tar = np.uint8(tar)
    for i in range(len(gen)):
        threshgen = cv2.cvtColor(gen[i], cv2.COLOR_BGR2GRAY)
        threshtar = cv2.cvtColor(tar[i], cv2.COLOR_BGR2GRAY)
        #print(np.unique(threshtar))
        arr = [0, 59, 104.5, 257]
        #arr = [0, 127.5, 256] #mean values based on np.unique(tar[i])
        #print(np.unique(gen[i]))
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
        #print(dice[i])
    return np.nanmean(dice)


def split(vid1, vid2):
    train = list()
    train1 = list()
    train2 = list()
    for filename in listdir(vid1):
        pixels = load_img(vid1 + filename)
        pixels = img_to_array(pixels)
        train.append(pixels)
        train1.append(pixels)

    random.shuffle(train1)

    for filename in listdir(vid2):
        pixels = load_img(vid2 + filename)
        pixels = img_to_array(pixels)
        train2.append(pixels)
        train.append(pixels)

    random.shuffle(train2)
    x = int(len(train1)*0.8)
    y = int(len(train2)*0.8)
    src_list, tar_list = list(), list()
    src_list1, tar_list1 = list(), list()
    for i in range(x):
        src_list.append(train1[i][:, :256])
        tar_list.append(train1[i][:, 256:])
    for i in range(y):
        src_list.append(train2[i][:, :256])
        tar_list.append(train2[i][:, 256:])

    for i in range(len(train)):
        src_list1.append(train[i][:, :256])
        tar_list1.append(train[i][:, 256:])


    src_list = np.asarray(src_list)
    tar_list = np.asarray(tar_list)
    src_list1 = np.asarray(src_list1)
    tar_list1 = np.asarray(tar_list1)
    np.savez_compressed('3class_train_BPonly.npz', src_list, tar_list)
    np.savez_compressed('3class_all_images_BPonly.npz', src_list1, tar_list1)
    print(src_list.shape)
    print(src_list1.shape)
    print(tar_list1.shape)


    src_list, tar_list = list(), list()
    for i in range(x, len(train1)):
        src_list.append(train1[i][:, :256])
        tar_list.append(train1[i][:, 256:])
    for i in range(y, len(train2)):
        src_list.append(train2[i][:, :256])
        tar_list.append(train2[i][:, 256:])

    src_list = np.asarray(src_list)
    tar_list = np.asarray(tar_list)
    np.savez_compressed('3class_test_ONLYBP.npz', src_list, tar_list)
    print(tar_list.shape)

#concatenate('thoracic/images/', 'thoracic/images/', 'thoracic/concatenated/', 1000)
#split('WithBeddingPlane/Vid1/beddingplane/', 'WithBeddingPlane/Vid2/beddingplane/')
test("")
#create_npz('thoracic/concatenated/', 'thoracic_multi.npz')
#test_rotated_blurred('3class_all_images.npz', 1000, " mulit 20 degree tilt", 20)

#test_rotated_blurred('3class_all_images.npz', 1000, " mulit 30 degree tilt", 30)

#test_rotated_blurred('3class_all_images.npz', 1000, " mulit 45 degree tilt", 45)
