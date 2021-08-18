import xml.etree.ElementTree as ET
import cv2
def parse_annotate(xmlpath, imgdirec):
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    print(root[4])
    count = 0
    scalingw = 1920/256
    scalingh = 1080/256
    for child in root[4]:
        points = child.get('points').split(';')
        first = points[0].split(',')
        second = points[1].split(',')
        first = [float(first[0])/scalingw, float(first[1])/scalingh]
        second = [float(second[0]) / scalingw, float(second[1]) / scalingh]
        image = cv2.imread('WithBeddingPlane/Vid2/annotated_mask_multiclass/mask_' + str(count).zfill(6) + '.PNG')
        image = cv2.line(image, (int(first[0]), int(first[1])), (int(second[0]), int(second[1])), (255, 255, 255))
        cv2.imwrite('WithBeddingPlane/Vid2/annotated_mask_multiclass/mask_' + str(count).zfill(6) + '.PNG', image)
        count += 1
        print(count)
    print(count)


parse_annotate('WithBeddingPlane/Vid2/annotations.xml', '')