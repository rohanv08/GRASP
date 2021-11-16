import pandas as pd
import numpy as np
import cv2
from keras_preprocessing.image import load_img, img_to_array
from os import path
import json
from os import path as pt
from matplotlib import pyplot

def scale_points(path, size=(256, 256)):
    width, height = size
    with open(path + 'instances_default.json') as f:
      data = json.load(f)

    scaling_width = data['images'][0]['width']/width
    scaling_height = data['images'][0]['height']/height

    for i in range(len(data['images'])):
        data['images'][i]['width'] = width
        data['images'][i]['height'] = height

    for i in range(len(data['annotations'])):
        for j in range(len(data['annotations'][i]['segmentation'][0])):
            if j % 2 == 0:
                data['annotations'][i]['segmentation'][0][j] /= scaling_width
            else:
                data['annotations'][i]['segmentation'][0][j] /= scaling_height

        for j in range(4):
            if j % 2 == 0:
                data['annotations'][i]['bbox'][j] /= scaling_width
            else:
                data['annotations'][i]['bbox'][j] /= scaling_height

        data['annotations'][i]['area'] /= (scaling_width*scaling_height)


    with open(path + 'annotations_scaled.json', 'w') as f:
      json.dump(data, f)


path = 'WithBeddingPlane/Vid1/annotations_scaled.json'

def annotate(path):
    with open(path + 'annotations_scaled.json') as f:
        data = json.load(f)
    print(len(data['images']))
    for i in range(len(data['images'])):
        print(i)
        print(data['images'][i]['file_name'])
        image = cv2.imread(path + 'frames/' + data['images'][i]['file_name'])
        anns = data['annotations'][i]
        points = anns['segmentation'][0]
        points = np.asarray(points)
        print(points)
        pts = []
        for j in range(int(len(points)/2)):
            pts.append([int(points[2*j]), int(points[2*j+1])])
        pts = np.asarray(pts)


        for j in range(int(len(points)/2)):
            if j == int(len(points)/2) - 1:
                image = cv2.line(image, (int(points[2*j]), int(points[2*j+1])),
                                 (int(points[0]), int(points[1])), (0, 0, 0), 1)
            else:
                image = cv2.line(image, (int(points[2*j]), int(points[2*j+1])),
                                 (int(points[2*(j+1)]), int(points[2*(j+1)+1])), (0, 0, 0), 1)

        #cv2.imshow('test', image)
        #cv2.waitKey(0)
        #new_image = np.zeros(image.shape, dtype="uint8")
        #cv2.fillPoly(image, np.array([pts], dtype=np.int32), (0, 0, 0))
        #image = cv2.bitwise_or(new_image, image)
        cv2.imwrite(path + '/outcropB&W/' + data['images'][i]['file_name'], image)

def understand_data(path):
    with open(path) as f:
      data = json.load(f)
      #print(data)

    #for i in range(len(data['images'])):
    print(data['images'][0])
    anns0 = data['annotations'][0]
    anns1 = data['annotations'][1]
    anns2 = data['annotations'][2]
    print(anns0)
    print(anns1)
    print(anns2)

def annotate_BP (path):
    src_list, tar_list = list(), list()
    for i in range(ran):
        filename = 'image_' + str(i).zfill(6) + '.PNG'
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)
        src_list.append(pixels[:, :256])
        tar_list.append(pixels[:, 256:])
#understand_data(path)
#scale_points('WithBeddingPlane/Vid1/')
annotate('WithBeddingPlane/Vid1/')

