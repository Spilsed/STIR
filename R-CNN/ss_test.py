import cv2
import time
import random

image = cv2.imread("img.png")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()

rects = ss.process()
rect = image[rects[0][1]:rects[0][1]+rects[0][3], rects[0][0]:rects[0][0]+rects[0][2]]
cv2.imshow("2", rect)
cv2.waitKey(0)

for i in range(0, len(rects), 100):
    output = image.copy()
    for (x, y, w, h) in rects[i:i + 100]:
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.imshow("Output", output)
    cv2.waitKey(0)