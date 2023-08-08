import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import numpy as np
from PIL import Image


#image = cv2.imread("img.png")
#resized = cv2.resize(image, (224, 244))

# ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
# ss.setBaseImage(image)
# ss.switchToSelectiveSearchFast()

class VGGBlock(nn.Module):
    def __init__(self, input, output, num_conv):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_conv):
            layers.append(nn.Conv2d(input, output, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            input = output
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),
            VGGBlock(64, 128, 2),
            VGGBlock(128, 256, 3),
            VGGBlock(256, 512, 3),
            VGGBlock(512, 512, 3)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512*49, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        for name, layer in self.features.named_modules():
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(add_hook())

        self.avgpool.register_forward_hook(add_hook())

        self.flatten.register_forward_hook(add_hook())

        for name, layer in self.classifier.named_modules():
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(add_hook())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class SVM(nn.Module):
    def __init__(self, classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(4096, classes)

        self.fc.register_forward_hook(add_hook())

    def forward(self, x):
        return self.fc(x)

class RCNN(nn.Module):
    def __init__(self, classes):
        super(RCNN, self).__init__()
        self.vgg = pretrained_custom_vgg16()
        self.svm = SVM(classes)

    def forward(self, x):
        x = self.vgg(x)
        return self.svm(x)

def add_hook():
    def size_hook(_model, _input, output):
        pass
        #print("Output Shape :", list(output.shape))
    return size_hook

def test_rcnn():
    model = VGG16()
    model.eval()
    model(torch.randn(1, 3, 244, 244))

def pretrained_custom_vgg16():
    vgg16 = VGG16()
    vgg16.eval()

    vgg16_pretrained = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    vgg16_pretrained.eval()

    state_dict_pretrained = vgg16_pretrained.state_dict()
    state_dict_custom = vgg16.state_dict()

    for i in range(len(state_dict_custom.keys())):
        state_dict_custom[list(state_dict_custom.keys())[i]] = state_dict_pretrained[
            list(state_dict_pretrained.keys())[i]]

    vgg16.load_state_dict(state_dict_custom)

    return state_dict_custom

def name_to_label(name):
    labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
              "tvmonitor"]
    return labels.index(name)

def label_to_name(label):
    labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
              "tvmonitor"]
    return labels[label]

def get_image_box(image, box):
    roi = np.asarray(image)[box[3]:box[2], box[1]:box[0]]
    return Image.fromarray(roi)

def train_svm(vgg, svm, dataloader, optimizer, num_epochs):
    vgg.eval()
    svm.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for image, annotation in dataloader:
            optimizer.zero_grad()
            features = []
            labels = []
            boxes = []

            for object in annotation["annotation"]["object"]:
                labels.append(name_to_label(object["name"][0]))
                boxes.append([int(object["bndbox"]["xmax"][0]), int(object["bndbox"]["xmin"][0]), int(object["bndbox"]["ymax"][0]), int(object["bndbox"]["ymin"][0])])

            image = image.squeeze()

            image = to_pil_image(image)

            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((244, 244))
            ])

            plt.imshow(image)
            plt.title("Full Image")
            plt.show()

            i = 0
            for box in boxes:
                image_box = get_image_box(image, box)
                image_box = preprocess(image_box).unsqueeze(0)
                plt.imshow(to_pil_image(image_box.squeeze()))
                plt.title(label_to_name(labels[i]))
                plt.show()
                feature = vgg(image_box)
                features.append(feature)
                i += 1

            features = torch.cat(features, dim=0)
            labels = torch.tensor(labels)

            outputs = svm(features)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f}")


def main():
    model = RCNN(classes=20)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    voc_dataset = VOCDetection(root="../VOC2012", year="2012", image_set="train", download=False,
                               transform=transform)
    dataloader = DataLoader(voc_dataset, batch_size=1, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    vgg = VGG16()
    svm = SVM(20)

    vgg.load_state_dict(pretrained_custom_vgg16())

    train_svm(vgg, svm, dataloader, optimizer, num_epochs=5)

if __name__ == "__main__":
    main()

