"""
Handles dataset organization for dataloader.
Additionally used as a utility to check model accuracy.
"""

import csv
import os
import torch
import tiramisuModel.tiramisu as tiramisu
import tiramisuModel.jerrys_helpers as jerrys_helpers
import cv2
import numpy as np
from PIL import Image, ImageOps

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CarlaAliDataset(Dataset):
    """
    Classification data from CarlaAliDataset.
    Represented as tuples of 3 x 150 x 200 images and their vectors of data/labels
    """
    def __init__(self, dataset_path):

        self.csv_tuples = []
        self.dataset_path = dataset_path
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor()])

        # Extract data from csv.
        labels_path = os.path.join(dataset_path, "labels.csv")
        with open(labels_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                self.csv_tuples.append((row[0], row[1], row[2], row[3], row[4])) 
                # 0 is rgb fname, 1 is sem name, 2 is data/labels vector

        # Cut out the csv headers from extracted data.
        self.csv_tuples = self.csv_tuples[1:]


    def __len__(self):
        """
        Your code here
        returns length of dataset.
        """
        return len(self.csv_tuples)


    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """

        # All pairs of image and label are added to csv_tuples string list.
        # Grab the data vector from 2nd index, get only labels and change to floats
        data = [float(self.csv_tuples[idx][2]),float(self.csv_tuples[idx][3]), float(self.csv_tuples[idx][4])]

        border = (0, 150, 0, 0) # cut 0 from left, 30 from top, right, bottom

        # Rgb image as a tensor
        rgb_image = Image.open(os.path.join(self.dataset_path, self.csv_tuples[idx][0]))
        # rgb_image = ImageOps.crop(rgb_image, border)
        rgb_tensor = self.transform(rgb_image)

        # Sem image as an input tensor
        sem_image = Image.open(os.path.join(self.dataset_path, self.csv_tuples[idx][1]))
        # sem_image = ImageOps.crop(sem_image, border)
        sem_tensor = self.transform(sem_image)

        return rgb_tensor, sem_tensor, torch.tensor(data)


def load_data(dataset_path, num_workers=0, batch_size=128):
    """
    Driver function to create dataset and return constructed dataloader.
    """
    dataset = CarlaAliDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


class CarlaJerrySemanticDataset(Dataset):
    """
    Classification data from CarlaAliDataset that will return rgb, manual semantic, and uncertainty.
    Represented as tuples of 3 x 150 x 200 images and their vectors of data/labels
    """
    def __init__(self, dataset_path):

        self.csv_tuples = []
        self.dataset_path = dataset_path

        # --- Initalize Fcnet Semantic Segmentation Model ---
        self.semantic_uncertainty_model = tiramisu.FCDenseNet67(n_classes=23).to(device='cuda:0')
        self.semantic_uncertainty_model.float()
        jerrys_helpers.load_weights(self.semantic_uncertainty_model,'weights/weights67latest.th')

        # ---- Transforms ----
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor()])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            mean = [0.534603774547577, 0.570066750049591, 0.589080333709717],
            std = [0.186295211315155, 0.181921467185020, 0.196240469813347])])

        # Extract data from csv.
        labels_path = os.path.join(dataset_path, "labels.csv")
        with open(labels_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                self.csv_tuples.append((row[0], row[1], row[2], row[3], row[4])) 
                # 0 is rgb fname, 1 is sem name, 2 is data/labels vector

        # Cut out the csv headers from extracted data.
        self.csv_tuples = self.csv_tuples[1:]


    def __getSegmentedData__(self, rgb_image):
        # Get normalized RGB tensor
        normalized_image = self.normalize(rgb_image)
        rgb_input = torch.unsqueeze(normalized_image, 0)
        rgb_input = rgb_input.to(torch.device("cuda:0"))

        # Get semantic segmented raw output
        self.semantic_uncertainty_model.eval().to(device='cuda:0')
        model_output = self.semantic_uncertainty_model(rgb_input) #Put single image rgb in tensor and pass in
        raw_semantic = jerrys_helpers.get_predictions(model_output) #Gets an unlabeled semantic image (red one)
        rgb_semantic = jerrys_helpers.color_semantic(raw_semantic[0]) #gets color converted semantic (like our convert cityscape)
        
        #Convert Jerry model float64 input to uint8
        rgb_semantic = cv2.normalize(src=rgb_semantic, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ##### Get Semantic Image #######

        #### Get Uncertainty Image ######
        self.semantic_uncertainty_model.train().to(device='cuda:0')
        mc_results = []
        output = self.semantic_uncertainty_model(rgb_input).detach().cpu().numpy()
        output = np.squeeze(output)
        # RESHAPE OUTPUT BEFORE PUTTING IT INTO mc_results
        # reshape into (480000, 23)
        # then softmax it
        output = jerrys_helpers.get_pixels(output)
        output = jerrys_helpers.softmax(output)
        mc_results.append(output)
        
        # boom we got num_samples passes of a single img thru the NN
        # now we use those samples to make uncertainty maps  
        mc_results = [mc_results]
        aleatoric = jerrys_helpers.calc_aleatoric(mc_results)[0]
        aleatoric = np.reshape(aleatoric, (300, 300))
        aleatoric = cv2.normalize(src=aleatoric, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        aleatoric = cv2.merge((aleatoric,aleatoric,aleatoric))
        return rgb_semantic, aleatoric
    
    def __len__(self):
        """
        Your code here
        returns length of dataset.
        """
        return len(self.csv_tuples)


    def __getitem__(self, idx):
        """
        This whole thing needs to do process image with model
        return a tuple: rgb, semantic, uncertainty and label
        https://github.com/Ayaanfaraz/imitation-learning/blob/serverBranch/imitation_learning.py#LL214C13-L214C13
        """

        # All pairs of image and label are added to csv_tuples string list.
        # Grab the data vector from 2nd index, get only labels and change to floats
        data = [float(self.csv_tuples[idx][2]),float(self.csv_tuples[idx][3]), float(self.csv_tuples[idx][4])]

        # Rgb image as a tensor
        rgb_image = Image.open(os.path.join(self.dataset_path, self.csv_tuples[idx][0]))

        # Get semantic and uncertainty tensor in jerrys way and apply same final transforms on it.
        sem_image, uncertainty_image = self.__getSegmentedData__(rgb_image)

        # -- DEBUG, save uncertainty and sem visualizations ---
        cv2.imwrite("sem1.jpg", sem_image)
        cv2.imwrite("aleatoric1.jpg", uncertainty_image)
        rgb_image.save("rgb1.jpg")

        sem_tensor = self.transform(Image.fromarray(sem_image))
        uncertainty_tensor = self.transform(Image.fromarray(uncertainty_image))
        rgb_tensor = self.transform(rgb_image)

        return rgb_tensor, sem_tensor, uncertainty_tensor, torch.tensor(data)

def load_custom_data(dataset_path, num_workers=0, batch_size=128):
    """
    Driver function to create dataset and return constructed dataloader which returns manually segmented images and uncertainty.
    """
    dataset = CarlaJerrySemanticDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(yhat, y): # yhat is not labels it is prediction
    """
    Returns accuracy between true labels and predictions.
    """
    def rmse(y_tensor, yhat_tensor):
        return torch.sqrt(torch.mean(torch.pow((y_tensor - yhat_tensor), 2)))

    steer_rmse = throttle_rmse = brake_rmse = 0
    for j in range(len(y)):
        steer_rmse += rmse(y[j, 0], yhat[j, 0])
        throttle_rmse += rmse(y[j, 1], yhat[j, 1])
        brake_rmse += rmse(y[j, 2], yhat[j, 2])

    steer_rmse /= len(y)
    throttle_rmse /= len(y)
    brake_rmse /= len(y)

    accuracy_steer = round(1 - steer_rmse.item(), 3)
    accuracy_throttle = round(1 - throttle_rmse.item(), 3)
    accuracy_brake = round(1 - brake_rmse.item(), 3)
    accuracy_avg = round((accuracy_steer + accuracy_throttle + accuracy_brake) / 3, 3)
    return accuracy_steer, accuracy_throttle, accuracy_brake, accuracy_avg
