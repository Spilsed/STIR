import cv2

image = cv2.imread("img.png")
resized = cv2.resize(image, (224, 244))

cv2.imshow("Re", resized)
cv2.waitKey(0)