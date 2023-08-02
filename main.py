import torch
import torch.nn as nn

# Define VGG backbone
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # VGG16 layers (excluding fully connected layers)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Use the original weights of VGG16 pretrained on ImageNet
        # If you have the weights, you can load them here using `torch.load()`

    def forward(self, x):
        x = self.features(x)
        return x

# Define Region Proposal Network (RPN)
class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        return logits, bbox_reg

# Define RoI Pooling Layer
class RoIPooling(nn.Module):
    def __init__(self, output_size):
        super(RoIPooling, self).__init__()
        self.roi_pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, features, rois):
        # Combine region proposals (rois) with feature maps (features)
        # and apply RoI pooling
        return self.roi_pool(features, rois)

# Define Fully Connected layers for object detection
class FastRCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNN, self).__init__()
        self.fc1 = nn.Linear(in_channels, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the features
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred

# Define the R-CNN architecture using VGG backbone
class R_CNN(nn.Module):
    def __init__(self, num_classes, num_anchors=9):
        super(R_CNN, self).__init__()
        self.vgg = VGG()
        self.rpn = RPN(512, num_anchors)
        self.roi_pooling = RoIPooling(output_size=(7, 7))
        self.fast_rcnn = FastRCNN(7 * 7 * 512, num_classes)

    def forward(self, x, rois):
        features = self.vgg(x)
        rpn_logits, rpn_bbox_reg = self.rpn(features)
        pooled_features = self.roi_pooling(features, rois)
        cls_score, bbox_pred = self.fast_rcnn(pooled_features)
        return rpn_logits, rpn_bbox_reg, cls_score, bbox_pred

# Instantiate the R-CNN model with VGG backbone
num_classes = 20  # Replace with the actual number of classes in your dataset
model = R_CNN(num_classes=num_classes)
