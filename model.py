import torch
import torch.nn as nn
import cv2
import torchvision.models

image = cv2.imread("img.png")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()

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
            nn.Linear(25088, 4096),
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
        self.vgg = VGG16()
        self.svm = SVM(classes)

    def forward(self, x):
        x = self.vgg(x)
        return self.svm(x)

def add_hook():
    def size_hook(_model, _input, output):
        print("Output Shape :", list(output.shape))
    return size_hook

if __name__ == '__main__':
    vgg16 = VGG16()
    full = RCNN(10)

    full(torch.randn(1, 3, 244, 244))

    vgg16_pretrained = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

    before = vgg16.state_dict()

    state_dict_pretrained = vgg16_pretrained.state_dict()
    state_dict_custom = vgg16.state_dict()

    for i in range(len(state_dict_custom.keys())):
        state_dict_custom[list(state_dict_custom.keys())[i]] = state_dict_pretrained[list(state_dict_pretrained.keys())[i]]
        print(torch.equal(state_dict_custom[list(state_dict_custom.keys())[i]], state_dict_pretrained[list(state_dict_pretrained.keys())[i]]))

    vgg16.load_state_dict(state_dict_custom, strict=True)

    print("Weights changed :", not torch.equal(before[list(before.keys())[2]], vgg16.state_dict()[list(vgg16.state_dict().keys())[2]]))
