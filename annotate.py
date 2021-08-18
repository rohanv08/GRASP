import pandas as pd
import numpy as np
import cv2
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


path = 'data/1.2/'

def annotate(path):
    with open(path + 'annotations/annotations_1.1.json') as f:
        data = json.load(f)

    for i in range(len(data['images'])):
        print(data['images'][i]['file_name'])
        image = cv2.imread(path + data['images'][i]['file_name'])
        anns = data['annotations'][i]
        points = anns['segmentation'][0]
        points = np.asarray(points)

        pts = []
        for j in range(int(len(points)/2)):
            pts.append([int(points[2*j]), int(points[2*j+1])])
        pts = np.asarray(pts)


        for j in range(int(len(points)/2)):
            if j == int(len(points)/2) - 1:
                image = cv2.line(image, (int(points[2*j]), int(points[2*j+1])),
                                 (int(points[0]), int(points[1])), (0, 0, 0), 2)
            else:
                image = cv2.line(image, (int(points[2*j]), int(points[2*j+1])),
                                 (int(points[2*(j+1)]), int(points[2*(j+1)+1])), (0, 0, 0), 2)

        #cv2.imshow('test', image)
        #cv2.waitKey(0)
        new_image = np.zeros(image.shape, dtype="uint8")
        cv2.fillPoly(new_image, np.array([pts], dtype=np.int32), (255, 255, 255))
        image = cv2.bitwise_and(new_image, image)
        cv2.imwrite(path + '/annotated/' + data['images'][i]['file_name'], image)

def understand_data(path):
    with open('WithBeddingPlane/Video1/annotations_scaled.json') as f:
      data = json.load(f)
      #print(data)

    #for i in range(len(data['images'])):
    print(data['images'][0]['file_name'])
    anns0 = data['annotations'][0]
    anns1 = data['annotations'][1]
    anns2 = data['annotations'][2]
    print(anns0)
    print(anns1)
    print(anns2)

understand_data(path)
#annotate(path)
#scale_points('WithBeddingPlane/Video2/')
