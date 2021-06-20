import cv2 as cv2
from PIL import Image as im

vid = cv2.VideoCapture('vids/1.1.mp4')
success, image = vid.read()
count = 0
while success:
  image = cv2.resize(image, (256, 256))
  #print(image.shape)
  cv2.imwrite("data/1.1/frame_" + str(count).zfill(6) + ".PNG", image)
  success, image = vid.read()
  print('Saved image ', count)
  count += 1


