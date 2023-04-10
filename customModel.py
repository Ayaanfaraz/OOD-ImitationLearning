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

        x = self.linear(F.relu(x)) # 64 output of features
        return x

        # Inspiration taken from following URLs:
        # https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
        # https://www.kaggle.com/code/reukki/pytorch-cnn-tutorial-with-cats-and-dogs

def save_model(model):
    from torch import save
    from os import path
    #if isinstance(model, CNNClassifier):
    return save(model.state_dict(), path.join("weights", 'fusionModel.th'))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join("weights", 'fusionModel.th'), map_location='cuda:1'))
    return r

class FusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = CNNClassifier()
        self.cnn2 = CNNClassifier()
        self.fc = torch.nn.Linear(128, 3)

    def forward(self, rgb, semantic):
        """
        @rgb: torch.Tensor((B,64))
        @semantic: torch.Tensor((B,64))
        @return: torch.Tensor((B,3))
        """

        rgb_features = self.cnn1(rgb)
        sem_features = self.cnn2(semantic)

        # Cat 64 and 64 features and otput steer, throttle, brake.
        x = torch.cat((rgb_features, sem_features), dim=1)
        x = self.fc(x)

        return x.view(-1, 3, 1)

