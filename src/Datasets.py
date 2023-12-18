import random
import torch
import torch.utils.data
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, ToPILImage
from utils import *


class X_CIFAR(torch.utils.data.Dataset):
    def __init__(self, data_samples, size=(32, 32), dataset='CIFAR', number_samples=60000,  increment_dif=False, FF_rep=False):

        self.data_samples = data_samples
        self.size = size
        self.number_samples = number_samples  # how train data to create many to create
        self.increment_dif = increment_dif
        self.normal_imgs, self.y_pos = self.generate_data()
        self.FF_rep = FF_rep
        self.dataset = dataset

    def resize_images(self, digit1, digit2):

        # Resize images to the same size
        digit1 = cv2.resize(digit1, self.size)
        digit2 = cv2.resize(digit2, self.size)

        return digit1, digit2


    def visualize(self, digit1, digit2, mask, hybrid):

        # Plot the original images, the binary mask, and the hybrid images
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))
        ax1.imshow(digit1, cmap='gray')
        ax1.set_title('Digit 1')
        ax2.imshow(digit2, cmap='gray')
        ax2.set_title('Digit 2')
        ax3.imshow(mask, cmap='gray')
        ax3.set_title('Binary Mask')
        ax4.imshow(hybrid, cmap='gray')
        ax4.set_title('Hybrid 1')
        plt.show()

        return

    def generate_data(self):

        normal_imgs = []
        y_pos = []

        for i in range(self.number_samples):

            idx1 = i
            y = self.data_samples[idx1][1]
            digit1 = np.array(self.data_samples[idx1][0])
            normal_imgs.append(digit1)
            y_pos.append(y)

        return normal_imgs, y_pos

    def __getitem__(self, index):

        # CIFAR100 experiment.
        # Following standard practice, normalized the channels by ((0.5074,0.4867,0.4411) and  (0.2011,0.1987,0.2025)
        if self.dataset == 'CIFAR100':
            transform = Compose([
                ToTensor(),
                Normalize(mean=[0.5074, 0.4867, 0.4411],
                                     std=[0.2011, 0.1987, 0.2025])])
        else:
            transform = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        # Load Positive and Negative Samples
        x_pos = transform(self.normal_imgs[index])
        y_pos = self.y_pos[index]

        # Transform Positive and Negative Samples
        x_pos = x_pos.reshape(3, self.size[1], self.size[0])  #
        y_pos = torch.tensor(np.asarray(y_pos)).long()

        if self.FF_rep:
            x_pos = x_pos.flatten()

        return x_pos, y_pos

    def __len__(self):
        return len(self.normal_imgs)


class X_MNIST(torch.utils.data.Dataset):
    def __init__(self, data_samples, size=(28, 28), number_samples=60000, dataset='MNIST',  increment_dif=False, FF_rep=False):

        self.data_samples = data_samples
        self.size = size
        self.repeats = 20
        self.threshold = 0.2
        self.threshold_decay = 0.95
        self.number_samples = number_samples  # how train data to create many to create
        self.increment_dif = increment_dif
        self.dataset = dataset
        self.normal_imgs, self.y_pos = self.generate_data()
        self.FF_rep = FF_rep

    def resize_images(self, digit1, digit2):

        # Resize images to the same size
        digit1 = cv2.resize(digit1, self.size)
        digit2 = cv2.resize(digit2, self.size)

        return digit1, digit2


    def visualize(self, digit1, digit2, mask, hybrid):

        # Plot the original images, the binary mask, and the hybrid images
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))
        ax1.imshow(digit1, cmap='gray')
        ax1.set_title('Digit 1')
        ax2.imshow(digit2, cmap='gray')
        ax2.set_title('Digit 2')
        ax3.imshow(mask, cmap='gray')
        ax3.set_title('Binary Mask')
        ax4.imshow(hybrid, cmap='gray')
        ax4.set_title('Hybrid 1')
        plt.show()

        return

    def generate_data(self):

        normal_imgs = []
        y_pos = []

        for i in range(self.number_samples):

            idx1 = i

            y = self.data_samples[idx1][1]

            digit1 = np.array(self.data_samples[idx1][0])

            normal_imgs.append(digit1)
            y_pos.append(y)

        return normal_imgs, y_pos

    def __getitem__(self, index):

        if self.dataset == 'MNIST':
            transform = Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,))])
        else:
            transform = Compose([
                ToTensor(),
                Normalize((0.5,), (0.5,))])

        # Load Positive and Negative Samples
        x_pos = transform(self.normal_imgs[index])
        y_pos = self.y_pos[index]

        # Transform Positive and Negative Samples
        x_pos = x_pos.reshape(1, self.size[1], self.size[0])  #

        y_pos = torch.tensor(np.asarray(y_pos)).long()

        if self.FF_rep:
            x_pos = x_pos.flatten()

        return x_pos, y_pos

    def __len__(self):
        return len(self.normal_imgs)
