import os
import random
import statistics
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image
import clip
import hashlib
from tqdm import tqdm


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
            nn.Linear(512 * 49, 4096),
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
        self.fc = nn.Linear(512, classes)

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
        # print("Output Shape :", list(output.shape))

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
    return float(labels.index(name))


def label_to_name(label):
    labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
              "tvmonitor"]
    return labels[label]


def get_image_box(image, box):
    roi = np.asarray(image)[box[3]:box[2], box[1]:box[0]]
    return Image.fromarray(roi)


def train_svm(svm, dataloader, dataloader_val, optimizer, scheduler, num_epochs):
    start_time = str(datetime.timestamp(datetime.now()) * 1000)

    model, preprocess = clip.load("ViT-B/32", device="cuda")

    losses = []
    val_losses = []

    svm.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, labels) in tqdm(enumerate(dataloader)):
            zero_grad = datetime.now()
            optimizer.zero_grad()

            features_start = datetime.now()
            for i in range(len(images)):
                features = []

                feature = model.encode_image(images[i].cuda(device=0)).to(torch.float32)
                features.append(feature)

                features = torch.cat(features, dim=0)

                outputs = svm(features)
                loss = F.cross_entropy(outputs.cuda(device=0), labels[i].long().cuda(device=0))
                loss.backward()

            optimizer_step = datetime.now()
            optimizer.step()
            losses.append(loss.item())

            with open("WCLIP" + start_time + ".txt", "w") as f:
                f.write(str(losses))

            print("Total : ", datetime.now() - zero_grad)

            print(f"Loss: {loss.item():.4f}")

        svm.eval()

        current_val_loss = []
        for batch_idx, (images, labels) in tqdm(enumerate(dataloader_val)):
            validation = datetime.now()

            features_start = datetime.now()
            for i in range(len(images)):
                features = []

                feature = model.encode_image(images[i].cuda(device=0)).to(torch.float32)
                features.append(feature)

                features = torch.cat(features, dim=0)

                outputs = svm(features)
                loss = F.cross_entropy(outputs.cuda(device=0), labels[i].long().cuda(device=0))
                loss.backward()

            val_losses.append(loss.item())
            current_val_loss.append(loss.item)

            with open("WCLIP" + start_time + "_val.txt", "w") as f:
                f.write(str(val_losses))

            print("Val Total : ", datetime.now() - validation)

            print(f"Val Loss: {loss.item():.4f}")

        torch.save(svm.state_dict(), "./models/WCLIP_attempt" + str(len(os.listdir("./models"))) + "E" + str(epoch))
        scheduler.step()

    plt.plot(losses)
    plt.show()

    return svm


def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def cv_rect_to_pil_rect(cv_rect):
    pil_rect = [cv_rect[2] + cv_rect[0], cv_rect[0], cv_rect[3] + cv_rect[1], cv_rect[1]]
    return pil_rect


prev_return = []


def main():
    vgg = VGG16()
    svm = SVM(21).cuda(device=device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((244, 244), antialias=True),
    ])

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def transform(image, annotations):
        global prev_return

        image_hash = hashlib.md5(image.tobytes("hex", "rgb")).hexdigest()
        original_image = image.copy()

        if image_hash not in os.listdir("./cache"):
            image = np.array(image)
            image = image[:, :, ::-1]
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchQuality()
            rects = ss.process()
            with open("./cache/" + image_hash, "a") as f:
                for rect in rects:
                    for point in rect:
                        f.write(str(point) + ",")
                    f.write("\n")
        else:
            final_rect = []
            rects = open("./cache/" + image_hash, "r").read().split("\n")
            for rect in rects:
                final_rect.append(rect.split(",")[:-1])
                final_rect[-1] = [int(i) for i in final_rect[-1]]
            rects = final_rect[:-1]

        g_rects = []
        annotations = annotations["annotation"]["object"]

        for obj in annotations:
            g_rects.append([[int(obj["bndbox"]["xmin"]),
                             int(obj["bndbox"]["ymin"]),
                             int(obj["bndbox"]["xmax"]),
                             int(obj["bndbox"]["ymax"])],
                            name_to_label(obj["name"])])

        positive = []
        negative = []

        for rect in rects:
            rect = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]

            for g_rect in g_rects:
                iou = intersection_over_union(g_rect[0], rect)
                if iou > 0.5:
                    positive.append([rect, g_rect[1]])
                else:
                    negative.append([rect, g_rect[1]])

        final_rects = []
        labels = []

        if len(positive) == 0 and len(prev_return) > 0:
            return prev_return[0], prev_return[1]

        for i in range(32):
            if len(positive) != 1:
                random_index = random.randint(0, len(positive) - 1)
            else:
                random_index = 0

            sample = positive[random_index]
            if len(positive) > 1:
                del positive[random_index]

            final_rects.append(preprocess(get_image_box(original_image, cv_rect_to_pil_rect(sample[0]))))
            labels.append(sample[1])

        for i in range(96):
            if len(negative) != 1:
                random_index = random.randint(0, len(negative) - 1)
            else:
                random_index = 0

            sample = negative[random_index]
            if len(negative) > 1:
                del negative[random_index]
            processed_rect = preprocess(get_image_box(original_image, cv_rect_to_pil_rect(sample[0])))
            final_rects.append(processed_rect)
            labels.append(0.0)

        temp = list(zip(final_rects, labels))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        final_rects, labels = list(res1), list(res2)

        final_rects = torch.tensor(np.asarray(final_rects))
        labels = torch.tensor(np.asarray(labels))

        prev_return = [final_rects, labels]
        return final_rects, labels

    voc_dataset = VOCDetection(root="../VOC2012", year="2012", image_set="train", download=False,
                               transforms=transform)

    voc_dataset_val = VOCDetection(root="../VOC2012", year="2012", image_set="trainval", download=False,
                                   transforms=transform)

    dataloader = DataLoader(voc_dataset, batch_size=32, shuffle=True, pin_memory=True)
    dataloader_val = DataLoader(voc_dataset_val, batch_size=32, shuffle=True, pin_memory=True)

    optimizer = optim.AdamW(svm.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    svm = train_svm(svm, dataloader, dataloader_val, optimizer, scheduler, num_epochs=5).cuda(device=0)

    torch.save(svm.state_dict(), "./models/WCLIP_attempt" + str(len(os.listdir("./models"))))


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 0

    main()
