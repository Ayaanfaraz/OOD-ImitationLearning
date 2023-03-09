import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3),
            torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU()
        )

        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3),
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU()
        )

        self.conv_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU()
        )

        self.conv_block4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.Conv2d(128, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU()
        )

        self.linear = torch.nn.Linear(12800, 64)
        self.fc = torch.nn.Linear(64, 3)

    def forward(self, x):
        """
        @x: torch.Tensor((B,3,150,200))
        @return: torch.Tensor((B,3))
        """
        # Normalize images for CNN by using grayscale
        # augumentation = transforms.Compose([
        #     transforms.Grayscale(num_output_channels=1)])

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = x.reshape(x.size(0), -1)

        x = self.linear(F.relu(x))
        x = self.fc(x)
        return x.view(-1, 3, 1)

        # Inspiration taken from following URLs:
        # https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
        # https://www.kaggle.com/code/reukki/pytorch-cnn-tutorial-with-cats-and-dogs

def save_model(model):
    from torch import save
    from os import path
    #if isinstance(model, CNNClassifier):
    return save(model.state_dict(), path.join("weights", 'reshapedcnn.th'))
    #raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join("weights", 'reshapedcnn.th'), map_location='cuda:1'))
    return r
