import cv2 as cv2
from PIL import Image as im

vid = cv2.VideoCapture('WithBeddingPlane/Video2/2.mp4')
success, image = vid.read()
count = 0
while success:
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.resize(image, (256, 256))
  #print(image.shape)
  cv2.imwrite("WithBeddingPlane/Video2/frames/frame_" + str(count).zfill(6) + ".PNG", image)
  success, image = vid.read()
  print('Saved image ', count)
  count += 1


