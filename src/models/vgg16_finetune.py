"""
VGG16 fine-tuned with ImageNet pretrained weights for SVHN digit classification.
Final classifier head replaced to output num_classes (11: digits 0-9 + background).
All layers are unfrozen for full fine-tuning.
"""

import torch.nn as nn
from torchvision import models


def build_vgg16(num_classes=11, freeze_features=False):
    """
    Load VGG16 with pretrained ImageNet weights and replace the
    final fully-connected layer to output ``num_classes`` logits.

    Args:
        num_classes:      number of logits (11: digits 0–9 + background class)
        freeze_features:  if True, freeze convolutional feature layers
                          and only train the classifier head
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the final linear layer (4096 → num_classes)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model
