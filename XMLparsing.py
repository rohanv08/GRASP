import xml.etree.ElementTree as ET
import cv2
import numpy as np
def parse_annotate_bp(xmlpath, path):
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    print(root[2])
    count = 0
    scalingw = 1920/256
    scalingh = 1080/256
    for child in root[4]:
        points = child.get('points').split(';')
        first = points[0].split(',')
        second = points[1].split(',')
        first = [float(first[0])/scalingw, float(first[1])/scalingh]
        second = [float(second[0]) / scalingw, float(second[1]) / scalingh]
        image = cv2.imread(path + '/outcropB&W/frame_' + str(count).zfill(6) + '.PNG')
        #image =  np.zeros((256,256,3), np.uint8)
        image = cv2.line(image, (int(first[0]), int(first[1])), (int(second[0]), int(second[1])), (0, 0, 0), 3)
        cv2.imwrite(path + '/outcropB&Wbp/mask_' + str(count).zfill(6) + '.PNG', image)
        count += 1
        print(count)
    print(count)

def parse_annotate_sky(xmlpath, path):
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    print(root[2])
    count = 0
    scalingw = 1920/256
    scalingh = 1080/256
    for child in root[2]:
        frame = int(int(child.get('frame')))
        print(int(str(frame).zfill(6)))
        image = cv2.imread(path + 'frames/frame_' + str(frame).zfill(6) + '.PNG')
        if image is None:
            break
        print(path + 'frames/frame_' + str(int(child.get('frame'))/10).zfill(6) + '.PNG')
        points = child.get('points').split(';')
        pts = []
        #print(points)
        for i in range(len(points)):
            first = float(points[i].split(',')[0])
            second = float(points[i].split(',')[1])
            first /= scalingw
            second /= scalingh
            pts.append(first)
            pts.append(second)
        pts = np.asarray(pts)
        #print(pts)

        ptsmain = []
        for j in range(int(len(pts) / 2)):
            ptsmain.append([int(pts[2 * j]), int(pts[2 * j + 1])])
        ptsmain = np.asarray(ptsmain)

        for j in range(int(len(pts) / 2)):
            if j == int(len(pts) / 2) - 1:
                image = cv2.line(image, (int(pts[2 * j]), int(pts[2 * j + 1])),
                                 (int(pts[0]), int(pts[1])), (255, 255, 255), 1)
            else:
                image = cv2.line(image, (int(pts[2 * j]), int(pts[2 * j + 1])),
                                 (int(pts[2 * (j + 1)]), int(pts[2 * (j + 1) + 1])), (255, 255, 255), 1)


        #new_image = np.ones(image.shape, dtype="uint8")*255
        cv2.fillPoly(image, np.array([ptsmain], dtype=np.int32), (0, 0, 0))
        #image = cv2.bitwise_or(new_image, image)
        cv2.imwrite(path + '/outcropB&W/frame_' + str(frame).zfill(6) + '.PNG', image)
        count += 1
        print(count)
    print(count)

def parse_annotate_ground(xmlpath, path):
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    print(root[3])
    count = 0
    scalingw = 1920/256
    scalingh = 1080/256
    for child in root[3]:
        frame = int(int(child.get('frame')))
        print(int(str(frame).zfill(6)))
        image = cv2.imread(path + 'outcropB&W/frame_' + str(frame).zfill(6) + '.PNG')
        if image is None:
            break
        print(path + 'frames/frame_' + str(int(child.get('frame'))/10).zfill(6) + '.PNG')
        points = child.get('points').split(';')
        pts = []
        #print(points)
        for i in range(len(points)):
            first = float(points[i].split(',')[0])
            second = float(points[i].split(',')[1])
            first /= scalingw
            second /= scalingh
            pts.append(first)
            pts.append(second)
        pts = np.asarray(pts)
        #print(pts)

        ptsmain = []
        for j in range(int(len(pts) / 2)):
            ptsmain.append([int(pts[2 * j]), int(pts[2 * j + 1])])
        ptsmain = np.asarray(ptsmain)

        for j in range(int(len(pts) / 2)):
            if j == int(len(pts) / 2) - 1:
                image = cv2.line(image, (int(pts[2 * j]), int(pts[2 * j + 1])),
                                 (int(pts[0]), int(pts[1])), (255, 255, 255), 1)
            else:
                image = cv2.line(image, (int(pts[2 * j]), int(pts[2 * j + 1])),
                                 (int(pts[2 * (j + 1)]), int(pts[2 * (j + 1) + 1])), (255, 255, 255), 1)


        #new_image = np.ones(image.shape, dtype="uint8")*255
        cv2.fillPoly(image, np.array([ptsmain], dtype=np.int32), (0, 0, 0))
        #image = cv2.bitwise_or(new_image, image)
        cv2.imwrite(path + '/outcropB&W/frame_' + str(frame).zfill(6) + '.PNG', image)
        count += 1
        print(count)
    print(count)


#parse_annotate_sky('WithBeddingPlane/Vid2/annotations.xml', 'WithBeddingPlane/Vid2/')
#parse_annotate_ground('WithBeddingPlane/Vid2/annotations.xml', 'WithBeddingPlane/Vid2/')
parse_annotate_bp('WithBeddingPlane/Vid1/annotations.xml', 'WithBeddingPlane/Vid1/')
