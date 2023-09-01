import random
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


# image = cv2.imread("img.png")
# resized = cv2.resize(image, (224, 244))

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


def train_svm(vgg, svm, dataloader, optimizer, num_epochs):
    for param in vgg.parameters():
        param.requires_grad = False

    svm.train()

    losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            for i in range(len(images)):
                features = []

                feature = vgg(images[i])
                features.append(feature)

                print(feature.dtype)

                features = torch.cat(features, dim=0)

                outputs = svm(features)
                print(outputs.shape)
                print(labels[i].shape)
                loss = F.cross_entropy(outputs, labels[i].long())
                loss.backward()
            optimizer.step()

            losses.append(loss)
            print(f"Loss: {loss.item():.4f}")

    plt.plot(losses)
    plt.show

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
    svm = SVM(21)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((244, 244), antialias=True),
    ])

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def transform(image, annotations):
        global prev_return

        original_image = image.copy()
        image = np.array(image)
        image = image[:, :, ::-1]
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchQuality()
        rects = ss.process()
        g_rects = []

        # for i in range(0, len(rects), 100):
        #     output = image.copy()
        #     for (x, y, w, h) in rects[i:i + 100]:
        #         color = [random.randint(0, 255) for j in range(0, 3)]
        #         cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        #     cv2.imshow("Output", output)
        #     cv2.waitKey(0)

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

                    # output = image.copy()
                    # cv2.rectangle(output, (rect[0], rect[1]), (rect[2], rect[3]), [0, 0, 255], 2)
                    # cv2.rectangle(output, (g_rect[0][0], g_rect[0][1]), (g_rect[0][2], g_rect[0][3]), [0, 255, 0], 2)
                    # print("POSITIVE :", iou)
                    # cv2.imshow("Output", output)
                    # cv2.waitKey(0)

                else:
                    negative.append([rect, g_rect[1]])

                    # output = image.copy()
                    # cv2.rectangle(output, (rect[0], rect[1]), (rect[2], rect[3]), [0, 0, 255], 2)
                    # cv2.rectangle(output, (g_rect[0][0], g_rect[0][1]), (g_rect[0][2], g_rect[0][3]), [0, 255, 0], 2)
                    # print("NEGATIVE :", iou)
                    # cv2.imshow("Output", output)
                    # cv2.waitKey(0)

        final_rects = []
        labels = []

        print("POSITIVE :", len(positive))
        if len(positive) == 0 and len(prev_return) > 0:
            return prev_return[0], prev_return[1]

        for i in range(32):
            sample = random.sample(positive, 1)[0]
            plt.imshow(get_image_box(original_image, cv_rect_to_pil_rect(sample[0])))
            #plt.show()
            final_rects.append(preprocess(get_image_box(original_image, cv_rect_to_pil_rect(sample[0]))))
            labels.append(sample[1])

        for i in range(96):
            sample = random.sample(negative, 1)[0]
            final_rects.append(preprocess(get_image_box(original_image, cv_rect_to_pil_rect(sample[0]))))
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

    dataloader = DataLoader(voc_dataset, batch_size=1, shuffle=True)
    optimizer = optim.SGD(svm.parameters(), lr=0.001, momentum=0.9)

    vgg.load_state_dict(pretrained_custom_vgg16())

    svm = train_svm(vgg, svm, dataloader, optimizer, num_epochs=5)


if __name__ == "__main__":
    torch.device("cuda")

    main()
