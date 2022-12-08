import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy


class CNN(nn.Module):
    """Convolutional Neural Network.
    
    We provide a simple network with a Conv layer, followed by pooling,
    and a fully connected layer. Modify this to test different architectures,
    and hyperparameters, i.e. different number of layers, kernel size, feature
    dimensions etc.

    See https://pytorch.org/docs/stable/nn.html for a list of different layers
    in PyTorch.
    """

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Sequential(nn.Conv2d(3, 6, 5), nn.ReLU()),
        #     nn.MaxPool2d(8, 8),
        #     nn.Sequential(nn.Linear(6 * 5 * 5, 90), nn.ReLU(), nn.Dropout(p=0.4)),
        #     nn.Sequential(nn.Linear(90,90), nn.ReLU(), nn.Dropout(p=0.4)),
        #     nn.Linear(90,6)
        # )
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(8, 8)
        self.fc1 = nn.Linear(6 * 5 * 5, 90)
        self.fc2 = nn.Linear(90,90)
        self.fc3 = nn.Linear(90,6)
        self.drop = nn.Dropout(p=0.3)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Forward pass of network."""
        x = self.pool(self.activation(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x

def get_transforms_train():
    """Return the transformations applied to images during training.
    
    See https://pytorch.org/vision/stable/transforms.html for a full list of 
    available transforms.
    """
    N,M=2,10
    transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(),
        ]
    )
    return transform


def get_transforms_val():
    """Return the transformations applied to images during validation.

    Note: You do not need to change this function 
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                    std=[0.5,0.5,0.5])
        ]
    )
    return transform


def get_loss_function():
    """Return the loss function to use during training. We use
       the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(network, lr, momentum=0.9):
    """Return the optimizer to use during training.
    
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    return optim.Adam(network.parameters(), lr=lr, weight_decay=0.001)
