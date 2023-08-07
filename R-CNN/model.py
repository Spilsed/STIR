import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import torchvision
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torch.nn.functional as F


image = cv2.imread("img.png")
resized = cv2.resize(image, (224, 244))

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
        print("Output Shape :", list(output.shape))
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

    # print("Weights changed :",
    #       torch.equal(state_dict_pretrained[list(state_dict_pretrained.keys())[2]],
    #                   vgg16.state_dict()[list(vgg16.state_dict().keys())[2]]))

    return state_dict_custom

def train_svm(num_epochs):
    transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor()
    ])

    coco_dataset = CocoDetection(root='data/train2014', annFile='data/annotations/instances_train2014.json',
                                 transform=transform)
    dataloader = DataLoader(coco_dataset, batch_size=1, shuffle=True)

    vgg16 = VGG16()
    vgg16.eval()
    svm = SVM(91)
    svm.train()

    optimizer = optim.SGD(svm.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for images, targets in dataloader:
            optimizer.zero_grad()

            with torch.no_grad():
                print("Images shape:", images.shape)
                features = vgg16(images)

            svm_outputs = svm(features)

            svm_targets = torch.tensor([target['category_id'] for target in targets])
            print("SVM targets shape:", svm_targets.shape)

            valid_batch_indices = torch.arange(images.size(0))
            print("Valid batch indices:", valid_batch_indices)

            svm_outputs = svm_outputs
            svm_targets = svm_targets

            print("Target :", svm_targets)
            print("Output :", svm_outputs.shape)

            svm_targets = svm_targets.long()

            loss = F.cross_entropy(svm_outputs, svm_targets)

            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training finished!")

    torch.save(svm.state_dict(), 'svm_model.pth')

if __name__ == '__main__':
    _model = RCNN(classes=80)
    _model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)

    train_svm(80)
