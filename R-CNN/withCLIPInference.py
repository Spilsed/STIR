import torch
import torch.nn as nn
import clip
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

class SVM(nn.Module):
    def __init__(self, classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        return self.fc(x)

def load_model(path):
    svm = SVM(21)
    svm.load_state_dict(path)
    return svm

def selective_search(ss, image):
    image = np.array(image)
    image = image[:, :, ::-1]
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()

def get_clip_features(clip_model, preprocess, image):
    image = preprocess(image).unsqueeze(0).to("cuda").to(torch.float32)

    with torch.no_grad():
        image_features = clip_model.encode_image(image).to(torch.float32)

    return image_features

def svm_inference(svm, clip_features):
    with torch.no_grad():
        return svm.forward(clip_features.to("cuda"))

def get_image_box(image, box):
    roi = np.asarray(image)[box[3]:box[2], box[1]:box[0]]
    return Image.fromarray(roi)

def cv_rect_to_pil_rect(cv_rect):
    pil_rect = [cv_rect[2] + cv_rect[0], cv_rect[0], cv_rect[3] + cv_rect[1], cv_rect[1]]
    return pil_rect

def main():
    # Load SVM
    svm = load_model(torch.load("./models/WCLIP_attempt7E4")).to("cuda")
    svm.eval()
    print("SVM Loaded!")

    # Load image
    image = Image.open("./2012_004331.jpg")

    # Get selective search regions
    print("Starting selective search...")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    rects = selective_search(ss, image)
    print("Search complete!")

    # Get image features using CLIP
    print("Collecting features...")
    rect_features = []
    model, preprocess = clip.load("ViT-B/32", device="cuda", jit=True)
    for i in tqdm(range(len(rects))):
        image_features = get_clip_features(model, preprocess, get_image_box(image, cv_rect_to_pil_rect(rects[i])))
        rect_features.append(image_features)

    # Get image feature outputs from SVM
    print("Collecting outputs...")
    outputs = []
    for i in tqdm(range(len(rect_features))):
        outputs.append(svm_inference(svm, rect_features[i]))

    print("Finding largest...")
    c_image = cv2.imread("./2012_004331.jpg")
    largest = []
    for i in range(len(outputs)):
        index = list(outputs[i][0]).index(max(list(outputs[i][0])[1:]))
        value = max(list(outputs[i][0]))

        if not largest:
            largest = [value, index, i]
        elif value > largest[0]:
            largest = [value, index, i]

        if value >= 4.1:
            x, y, w, h = rects[i]
            cv2.rectangle(c_image, (x, y), (x + w, y + h), [150, 0, 150], 2)

    print("Value :", largest[0])
    print("Index :", largest[2], largest[1])

    get_image_box(image, cv_rect_to_pil_rect(rects[largest[2]])).show()
    cv2.imshow("final", c_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
