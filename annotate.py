import pandas as pd
import numpy as np
import cv2
from os import path
import json
from matplotlib import pyplot

csv = pd.read_csv('geology-exposed-classifications.csv')
csv = csv.filter(['workflow_name', 'annotations', 'subject_data'])
csv = csv[csv['workflow_name'] == 'Outcrop ID']
print(csv.shape[0])
write = 1

def outline_image(image, points):
 print ('k')

outcrop = 0
no_outcrop = 0
part_outcrop = 0

for i in range(csv.shape[0]):
    obj = json.loads(csv.iloc[i]['subject_data'])
    obj = obj[list(obj.keys())[0]]
    if path.exists('pics/' + obj['Filename']):
        image = cv2.imread('pics/' + obj['Filename'])
        original = cv2.imread('pics/' + obj['Filename'])
        annotations = json.loads(csv.iloc[i]['annotations'])
        #print(annotations)
        if annotations[0]['value'] == 'All rock wall outcrop':
            outcrop = outcrop + 1
            cv2.imwrite('annotated/labelled/' + str(write) + '.jpg', image)
            cv2.imwrite('annotated/original/' + str(write) + '.jpg', original)
            write = write + 1

        if annotations[0]['value'] == 'Part of the image shows an outcrop, ' \
                                      'but I can also see other things, such as ground, sky, or vegetation':
            part_outcrop = part_outcrop + 1
            if len(annotations) >= 2:
                val = annotations[1]['value']
                if (val != None and len(val) > 0):
                    points = val[0]['points']
                    pts = []
                    for j in range(len(points)):
                        pts.append([int(points[j]['x']), int(points[j]['y'])])
                    pts = np.asarray(pts)

                    for j in range(len(points)):
                        if j == len(points) - 1:
                            pt1 = (int(points[j]['x']), int(points[j]['y']))
                            pt2 = (int(points[0]['x']), int(points[0]['y']))
                            image = cv2.line(image, pt1, pt2, (0,  0, 0), 10)
                        else:
                            pt1 = (int(points[j]['x']), int(points[j]['y']))
                            pt2 = (int(points[j+1]['x']), int(points[j+1]['y']))
                            image = cv2.line(image, pt1, pt2, (0, 0, 0), 10)

                    new_image = np.zeros(image.shape, dtype="uint8")
                    cv2.fillPoly(new_image, [pts], (255, 255, 255))
                    #cv2.imshow('new', new_image)
                    #cv2.waitKey(0)
                    image = cv2.bitwise_and(new_image, image)
                    #cv2.imshow('image', image)
                    #cv2.waitKey(0)
                cv2.imwrite('annotated/labelled/' + str(write) + '.jpg', image)
                cv2.imwrite('annotated/original/' + str(write) + '.jpg', original)
                write = write + 1

        if annotations[0]['value'] == 'No outcrop- it is all something else, like ground or sky or plants.':
            no_outcrop = no_outcrop + 1




print(outcrop)
print(no_outcrop)
print(part_outcrop)





