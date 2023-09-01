import os
import hashlib
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

for image_file in tqdm(os.listdir("../VOC2012/VOCdevkit/VOC2012/JPEGImages")):
    image = Image.open("../VOC2012/VOCdevkit/VOC2012/JPEGImages/" + image_file)
    image_hash = hashlib.md5(image.tobytes("hex", "rgb")).hexdigest()
    if image_hash not in os.listdir("./cache"):
        image = np.array(image)
        image = image[:, :, ::-1]
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        with open("./cache/" + image_hash, "a") as f:
            for rect in rects:
                for point in rect:
                    f.write(str(point) + ",")
                f.write("\n")