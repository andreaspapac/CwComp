import random
import torch
import torch.utils.data
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, ToPILImage
import numpy as np
from utils import *
from Layer_cnn import *

''' This .py File contains extra code for the Older version that was not based on argparse commands.
    It is used only by the train_models_SFvsGF.py file that can run the Goodness Predictor instead 
    of the Global Averaging Predictor. This file contains:
    - Models ESANN and AAAI - For the Models presented in the corresponding Papers.
    - Conv_Old - Conv2d Class that works with the Previous Version of Models and with Gd Predictor.
    - Old Datasets Classes that also contain Methods and Different Options for the Creation of different types of 
    Negative Data and Distorted Images
    '''

# Old Conv2d Layer Class that works with the Previous Version of Models and with Gd Predictor
class Conv_Old(nn.Conv2d):
    def __init__(self, in_dims, in_channels=1, out_channels=8, num_classes=10, kernel_size=7, stride=1, padding=1,
                 maxpool=True, groups=1, droprate=0, loss_criterion='CwC_CE'):
        super(Conv_Old, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=groups)

        self.dims = in_dims
        self.outc = out_channels
        self.kernel_size = kernel_size
        self.outsize = int(((self.dims[1] - self.kernel_size + (2 * padding)) / stride) + 1)
        self.ismaxpool = maxpool
        self.groups = groups

        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.outsize = int(self.outsize / 2)  # if maxpool

        self.wsz = self.outsize  # for visualisation of features
        self.N_neurons_out = self.outc * self.outsize ** 2  # total number of output neurons
        self.next_dims = [self.outc, self.outsize, self.outsize]

        self.conv_bn = nn.BatchNorm2d(self.outc, eps=1e-4)
        self.relu = nn.ReLU()  # , nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropout = torch.nn.Dropout(p=droprate)
        self.lr = 0.01  # 0.1
        # self.opt = SGD(self.parameters(), lr=self.lr, momentum=0.2, weight_decay=1e-5)
        self.opt = Adam(self.parameters(), lr=self.lr)  # , weight_decay=1e-4
        self.threshold = 2.0

        self.loss_criterion = loss_criterion
        if self.loss_criterion == 'CwC':
            self.criterion = CwCLoss()
        elif self.loss_criterion == 'CwC_CE':
            # CwC with nn.CrossEntropyLoss(): the same with other inputs
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_criterion == 'PvN':
            self.criterion = PvNLoss()
        elif self.loss_criterion == 'CWG':
            self.criterion = CWGLoss(0.28)
        else:
            print('Not Valid Loss Function selected!')
            exit()

        self.ep_losses = []
        self.num_classes = num_classes

        print(self.dims, self.N_neurons_out)

    def forward(self, x):

        x = self.dropout(x)

        # Forward Pass
        x = self._conv_forward(x, self.weight, self.bias)
        x = self.relu(x)
        if self.ismaxpool:
            x = self.maxpool(x)
        x = self.conv_bn(x)
        y = x

        return y

    def setdropout(self, drop_rate):
        self.dropout = torch.nn.Dropout(p=drop_rate)

    def vis_features(self, features):
        # Plot features, Positive and Negative
        # Create a plot of the number of features positive and negative
        per_class = True

        if per_class:
            feature_sets = torch.split(features, int(self.outc / self.num_classes), dim=0)
            N_columns = len(feature_sets)


            fig, axs = plt.subplots(1, N_columns, figsize=(16, 16))
            # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 4, figsize=(8, 8))
            axs[int(N_columns / 2)].set_title('Features of Different Channel Blocks')

            for cb, channelblock in enumerate(feature_sets):
                cb_rep = channelblock.mean(0)
                axs[cb].imshow(cb_rep.detach().cpu(), cmap='gray')

            plt.show()

    def goodness_factorCW(self, y, gt):

        #                           # y  : R^BxCxHxW
        # Create Masks to Select Positive Channels and Negative Channels
        pos_mask = torch.zeros((gt.shape[0], self.num_classes), dtype=torch.uint8).cuda()

        for i, idx in enumerate(gt):
            pos_mask[i, idx] = 1

        pos_mask = pos_mask > 0.5
        neg_mask = ~pos_mask

        # Split activations by Number_of_classes sets
        y_sets = torch.split(y, int(self.outc / self.num_classes), dim=1)

        # Add goodnesses

        for i, y in enumerate(y_sets):
            gf_y = y.pow(2).mean((1, 2, 3))
            gf_y = torch.reshape(gf_y, (gf_y.shape[0], 1))
            if i == 0:
                gf = gf_y
            else:
                gf = torch.cat((gf, gf_y), 1)

        # gf: R^BxC
        g_pos = gf[pos_mask]  # gf+: R^Bx1
        g_neg = gf[neg_mask]  # gf-: R^Bx(C-1)

        g_neg = torch.reshape(g_neg, (gf.shape[0], gf.shape[1] - 1))
        # print(g_pos.shape, g_neg.shape)
        g_neg = g_neg.mean(1)  # gf-: R^Bx1

        return g_pos, g_neg, gf

    def lr_decay(self):
        # decrease learning rate if loss becomes low
        decay = 0.95
        self.lr *= decay
        self.opt = Adam(self.parameters(), lr=self.lr, betas=(0.95, 0.999))

    def epoch_loss(self):
        epl_mean = torch.tensor(self.ep_losses).mean().item()
        # if abs(epl_mean - self.ep_losses[-1]) < 0.00001:
        #     self.lr_decay()
        #     print('lr decay, new lr = ', self.lr)
        self.ep_losses = []
        # print(self.ep_losses)
        # print('ep losses', self.ep_losses)
        return epl_mean

    def forward_forward(self, x_pos, x_neg, gt, show):

        # forward pass
        y_pos = self.forward(x_pos)

        gt = gt.cuda()

        g_pos, g_neg, gf = self.goodness_factorCW(y_pos, gt)

        if self.loss_criterion == 'CwC':
            loss = self.criterion(g_pos, gf)
        elif self.loss_criterion == 'CwC_CE':
            loss = self.criterion(gf, gt)
        elif self.loss_criterion == 'PvN':
            loss = self.criterion(g_pos, g_neg)
        elif self.loss_criterion == 'CWG':
            loss = self.criterion(g_pos, g_neg, gf, gt)

        self.loss = loss
        self.ep_losses.append(loss)

        # if show:
        #     print('------------------------------')
        #     print('g_pos_CW: {}, g_neg_CW:{}'.format(g_pos.mean().item(), g_neg.mean().item()))

        self.opt.zero_grad()
        # this backward just compute the derivative and hence
        # is not considered backpropagation.
        loss.backward()
        self.opt.step()

        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    def ff_infer(self, x):
        # x = x / (LA.norm(x, 2, 1, keepdim=True) + 1e-4)
        x = self._conv_forward(x, self.weight, self.bias)
        x = self.relu(x)
        if self.ismaxpool:
            x = self.maxpool(x)
        y = self.conv_bn(x)
        # return self.forward(x).detach()
        return y.detach()

# # # Models # # #
# Adjusted Code for Model presented in ESANN 2023 Paper to have the extra components of AAAI version
class ESANN(torch.nn.Module):

    def __init__(self, batch_size, dataset='MNIST', ILT='Acc'):
        super().__init__()

        self.iter = 1
        self.batch_size = batch_size
        self.show_iters = 800

        self.nn_layers = []
        self.convb1_layers = []
        self.convb2_layers = []

        if dataset == 'MNIST':
            if ILT == 'Fast':
                self.start_end = [[0, 3], [1, 4], [2, 5], [3, 10], [4, 20]]
            else:
                self.start_end = [[0, 1], [0, 3], [0, 5], [0, 10], [0, 20]]
            CNN_l1_dims = [1, 28, 28]
        elif dataset == 'FMNIST':
            if ILT == 'Fast':
                self.start_end = [[0, 7], [1, 10], [2, 14], [3, 30], [4, 40]]
            else:
                self.start_end = [[0, 7], [0, 10], [0, 14], [0, 30], [0, 40]]
            CNN_l1_dims = [1, 28, 28]
        else:
            if ILT == 'Fast':
                self.start_end = [[0, 12], [2, 18], [4, 24], [6, 36], [7, 50]]
            else:
                self.start_end = [[0, 12], [0, 18], [0, 24], [0, 36], [0, 50]]
            CNN_l1_dims = [3, 32, 32]

        # # # CNN LAYERS # # #

        kernel_size = 3
        N_Classes = 10
        CNN_l1_outch = 20

        self.convb1_layers += [Conv_Old(CNN_l1_dims, in_channels=CNN_l1_dims[0], out_channels=CNN_l1_outch,
                                   kernel_size=kernel_size, maxpool=False, groups=1, droprate=0, loss_criterion='CWG').cuda()]

        CNN_l2_dims = self.convb1_layers[-1].next_dims
        self.convb1_layers += [Conv_Old(CNN_l2_dims, in_channels=20, out_channels=80, num_classes=N_Classes,
                                   kernel_size=kernel_size, maxpool=True, groups=1, droprate=0, loss_criterion='CWG').cuda()]

        CNN_l2_dims = self.convb1_layers[-1].next_dims
        self.convb1_layers += [Conv_Old(CNN_l2_dims, in_channels=80, out_channels=320, num_classes=N_Classes,
                                   kernel_size=kernel_size, maxpool=True, groups=1, droprate=0, loss_criterion='CWG').cuda()]

        # # # Fully Connected Layers
        self.FC_dims = [self.convb1_layers[-1].N_neurons_out, 1024, 1024]
        print(self.FC_dims)
        # #
        for d in range(len(self.FC_dims) - 1):
            self.nn_layers += [FC_LayerCW(self.FC_dims[d], self.FC_dims[d + 1], d,  droprate=0).cuda()]

        # # # # Softmax Classifiers
        self.classifier_b1 = Softmax_CF(self.convb1_layers[-1].N_neurons_out).cuda()

    def predict(self, x):
        goodness_per_label = []
        h = x
        for i, convlayer in enumerate(self.convb1_layers):
            h = convlayer.ff_infer(h)  # The hidden activities in all but not the CNN layers are used for Prediction

        b1_out = h.clone()

        nnh = h
        for label in range(10):
            h = overlay_y_on_x3d(nnh, label)
            goodness = []
            for i, nnlayer in enumerate(self.nn_layers):
                h = nnlayer(h)
                goodness += [h.pow(2).mean(1)]
                if i < len(self.nn_layers) - 1:
                    h = overlay_y_on_x(h, label)  # overlay label on smoothed layer and first linear relu

            goodness_per_label += [sum(goodness).unsqueeze(1)]  #[class_score.unsqueeze(1)]

        cl_1 = self.classifier_b1(b1_out)

        goodness_per_label = torch.cat(goodness_per_label, 1)
        class_score = torch.mul(goodness_per_label, cl_1)

        return goodness_per_label.argmax(1), class_score.argmax(1), cl_1.argmax(1)


    def train(self, x_pos, x_neg, y, y_n, epoch):

        start_end = self.start_end

        if self.iter % self.show_iters == 0:
            show = True
        else:
            show = False

        h_pos = x_pos
        h_neg = h_pos.clone()
        h_neg = channel_shuffle(h_neg, groups=random.randint(1, 20))

        for i, convlayer in enumerate(self.convb1_layers):
            s, e = start_end[i]
            if epoch in list(range(s, e)):
                h_pos, h_neg = convlayer.forward_forward(h_pos, h_neg, y.cuda(), show)
                if show:
                    print('Training ConvBlock: 1 - Layer: ', i, '...')
            else:
                h_pos = convlayer.ff_infer(h_pos)

            h_neg = h_pos.clone()
            h_neg = channel_shuffle(h_neg, groups=random.randint(1, 20))

        b1_out = h_pos.clone()      # Block 1 out

        for i, convlayer in enumerate(self.convb2_layers):
            s, e = start_end[i + len(self.convb1_layers)]
            if epoch in list(range(s, e)):

                h_pos, h_neg = convlayer.forward_forward(h_pos, h_neg, y.cuda(), show)

                if show:
                    print('Training ConvBlock: 2 - Layer: ', i, '...')

            else:
                h_pos = convlayer.ff_infer(h_pos)

            h_neg = h_pos.clone()
            h_neg = channel_shuffle(h_neg, groups=random.randint(1, 20))

        h_pos = overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
        h_neg = overlay_y_on_x3d(h_pos, y_n)

        for i, nnlayer in enumerate(self.nn_layers):
            s, e = start_end[i + len(self.convb1_layers) + len(self.convb2_layers)]
            if epoch in list(range(s, e)):
                h_pos, h_neg = nnlayer.forward_forward(h_pos, h_neg, y.cuda(), show)
                h_pos = overlay_y_on_x(h_pos, y)

                if show:
                    print('Training NN Layer', i, '...')
            else:
                h_pos = nnlayer(h_pos)
                h_neg = nnlayer(h_neg)

            rnd = random.randint(0, 1)
            if rnd == 1:
                h_neg = overlay_y_on_x(h_neg, y_n)
            else:
                h_neg = overlay_y_on_x(h_neg, y)

        if epoch >= start_end[-1][0]:
            pred_ = self.classifier_b1.train_classifier(b1_out, y.cuda(), show)

        self.iter += 1

    def combine_inputs(self, pred_block, b_out):
        b_out = b_out.reshape(b_out.size(0), -1)
        ratio = b_out.max(1).values[0] / pred_block.max(1).values[0]
        pred_block *= ratio
        sf_in = torch.cat((b_out, pred_block), 1)

        return sf_in

# Old Code for AAAI Model that runs Goodness Predictor (Gd) Instead of Global Averaging Predictor (GA)
class AAAI(torch.nn.Module):

    def __init__(self, batch_size, dataset='MNIST', ILT='Acc', loss_criterion='CwC_CE', CFSE=True):
        super().__init__()

        self.iter = 1
        self.batch_size = batch_size
        self.show_iters = 800

        self.nn_layers = []
        self.convb1_layers = []
        self.convb2_layers = []

        if dataset == 'MNIST':
            if ILT == 'Fast':
                self.start_end = [[0, 3], [1, 4], [2, 5], [3, 0], [4, 20], [5, 20]]
            else:
                self.start_end = [[0, 1], [0, 3], [0, 4], [0, 5], [0, 20], [0, 20]]
            CNN_l1_dims = [1, 28, 28]
        elif dataset == 'FMNIST':
            if ILT == 'Fast':
                self.start_end = [[0, 7], [1, 10], [2, 13], [3, 16], [4, 30], [5, 40]]
            else:
                self.start_end = [[0, 7], [0, 10], [0, 13], [0, 16], [0, 30], [0, 40]]
            CNN_l1_dims = [1, 28, 28]
        else:
            if ILT == 'Fast':
                self.start_end = [[0, 11], [2, 18], [4, 26], [6, 32], [8, 36], [10, 50]]
            else:
                self.start_end = [[0, 11], [0, 16], [0, 21], [0, 26], [0, 36], [0, 50]]
            CNN_l1_dims = [3, 32, 32]

        # # # CNN LAYERS # # #

        kernel_size = 3
        N_Classes = 10
        CNN_l1_outch = 20

        if CFSE:
            groups = [1, 10, 1, 10]
        else:
            groups = [1, 1, 1, 1]


        self.convb1_layers += [Conv_Old(CNN_l1_dims, in_channels=CNN_l1_dims[0], out_channels=CNN_l1_outch,
                                   kernel_size=kernel_size, maxpool=False, groups=groups[0], droprate=0, loss_criterion=loss_criterion).cuda()]

        CNN_l2_dims = self.convb1_layers[-1].next_dims
        self.convb1_layers += [Conv_Old(CNN_l2_dims, in_channels=20, out_channels=80, num_classes=N_Classes,
                                        kernel_size=kernel_size, maxpool=True, groups=groups[1], droprate=0, loss_criterion=loss_criterion).cuda()]

        CNN_l2_dims = self.convb1_layers[-1].next_dims
        self.convb1_layers += [Conv_Old(CNN_l2_dims, in_channels=80, out_channels=240, num_classes=N_Classes,
                                   kernel_size=kernel_size, maxpool=False, groups=groups[2], droprate=0, loss_criterion=loss_criterion).cuda()]

        CNN_l2_dims = self.convb1_layers[-1].next_dims
        self.convb1_layers += [Conv_Old(CNN_l2_dims, in_channels=240, out_channels=480, num_classes=N_Classes,
                                   kernel_size=kernel_size, maxpool=True,  groups=groups[3], droprate=0, loss_criterion=loss_criterion).cuda()]

        # # # Fully Connected Layers
        self.FC_dims = [self.convb1_layers[-1].N_neurons_out, 1024, 1024]
        print(self.FC_dims)
        # #
        for d in range(len(self.FC_dims) - 1):
            self.nn_layers += [FC_LayerCW(self.FC_dims[d], self.FC_dims[d + 1], d,  droprate=0).cuda()]

        # # # # Softmax Classifiers
        self.classifier_b1 = Softmax_CF(self.convb1_layers[-1].N_neurons_out).cuda()

    def predict(self, x):
        goodness_per_label = []
        h = x
        for i, convlayer in enumerate(self.convb1_layers):
            h = convlayer.ff_infer(h)  # The hidden activities in all but not the CNN layers are used for Prediction

        b1_out = h.clone()

        nnh = h
        for label in range(10):
            h = overlay_y_on_x3d(nnh, label)
            goodness = []
            for i, nnlayer in enumerate(self.nn_layers):
                h = nnlayer(h)
                goodness += [h.pow(2).mean(1)]
                if i < len(self.nn_layers) - 1:
                    h = overlay_y_on_x(h, label)  # overlay label on smoothed layer and first linear relu

            goodness_per_label += [sum(goodness).unsqueeze(1)]  #[class_score.unsqueeze(1)]

        cl_1 = self.classifier_b1(b1_out)

        goodness_per_label = torch.cat(goodness_per_label, 1)
        class_score = torch.mul(goodness_per_label, cl_1)

        return goodness_per_label.argmax(1), class_score.argmax(1), cl_1.argmax(1)


    def train(self, x_pos, x_neg, y, y_n, epoch):

        start_end = self.start_end

        if self.iter % self.show_iters == 0:
            show = True
        else:
            show = False

        h_pos = x_pos
        h_neg = h_pos.clone()
        h_neg = channel_shuffle(h_neg, groups=random.randint(1, 20))

        for i, convlayer in enumerate(self.convb1_layers):
            s, e = start_end[i]
            if epoch in list(range(s, e)):
                h_pos, h_neg = convlayer.forward_forward(h_pos, h_neg, y.cuda(), show)
                if show:
                    print('Training ConvBlock: 1 - Layer: ', i, '...')
            else:
                h_pos = convlayer.ff_infer(h_pos)

            h_neg = h_pos.clone()
            h_neg = channel_shuffle(h_neg, groups=random.randint(1, 20))

        b1_out = h_pos.clone()      # Block 1 out

        for i, convlayer in enumerate(self.convb2_layers):
            s, e = start_end[i + len(self.convb1_layers)]
            if epoch in list(range(s, e)):

                h_pos, h_neg = convlayer.forward_forward(h_pos, h_neg, y.cuda(), show)

                if show:
                    print('Training ConvBlock: 2 - Layer: ', i, '...')

            else:
                h_pos = convlayer.ff_infer(h_pos)

            h_neg = h_pos.clone()
            h_neg = channel_shuffle(h_neg, groups=random.randint(1, 20))

        h_pos = overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
        h_neg = overlay_y_on_x3d(h_pos, y_n)

        for i, nnlayer in enumerate(self.nn_layers):
            s, e = start_end[i + len(self.convb1_layers) + len(self.convb2_layers)]
            if epoch in list(range(s, e)):
                h_pos, h_neg = nnlayer.forward_forward(h_pos, h_neg, y.cuda(), show)
                h_pos = overlay_y_on_x(h_pos, y)

                if show:
                    print('Training NN Layer', i, '...')
            else:
                h_pos = nnlayer(h_pos)
                h_neg = nnlayer(h_neg)

            rnd = random.randint(0, 1)
            if rnd == 1:
                h_neg = overlay_y_on_x(h_neg, y_n)
            else:
                h_neg = overlay_y_on_x(h_neg, y)

        if epoch >= start_end[-1][0]:
            pred_ = self.classifier_b1.train_classifier(b1_out, y.cuda(), show)

        self.iter += 1

    def combine_inputs(self, pred_block, b_out):
        b_out = b_out.reshape(b_out.size(0), -1)
        ratio = b_out.max(1).values[0] / pred_block.max(1).values[0]
        pred_block *= ratio
        sf_in = torch.cat((b_out, pred_block), 1)

        return sf_in


# # #  Datasets # # #
# Old Datasets Classes that also contain Methods and Different Options
# for the Creation of different types of Negative Data Distorted Images
class Hybrid_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, data_samples, size=(32, 32), number_samples=60000, increment_dif=False):

        self.data_samples = data_samples
        self.size = size
        self.repeats = 20
        self.threshold = 0.2
        self.threshold_decay = 0.95
        self.number_samples = number_samples  # how train data to create many to create
        self.increment_dif = increment_dif

        # Define a custom filter
        filter_kernel1 = np.array([
            [0, 1, 2, 1, 0],
            [1, 2, 4, 2, 1],
            [2, 4, 10, 4, 2],
            [1, 2, 4, 2, 1],
            [0, 1, 2, 1, 0]], dtype=np.float32) / 50  # Normalize the kernel

        filter_kernel2 = np.array([
            [1, 2, 1]
        ], dtype=np.float32) / 4  # Normalize the kernel

        self.filters = [filter_kernel1, filter_kernel2]

        # create random masks
        # self.masks = self.random_regions_mask()
        # generate data
        self.normal_imgs, self.hybrid_imgs, self.y_pos, self.y_neg = self.generate_data()

    def resize_images(self, digit1, digit2):

        # Resize images to the same size
        digit1 = cv2.resize(digit1, self.size)
        digit2 = cv2.resize(digit2, self.size)

        return digit1, digit2

    def visualize(self, img1, img2, hybrid):

        # Plot the original images, the binary mask, and the hybrid images
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 5))
        ax1.imshow(img1)
        ax1.set_title('Image 1')
        ax2.imshow(img2)
        ax2.set_title('Image 2')
        ax3.imshow(hybrid)
        ax3.set_title('Hybrid')
        plt.show()
        return

    def generate_data(self):

        hybrid_imgs = []
        y_negs = []
        normal_imgs = []
        y_pos = []

        for i in range(self.number_samples):

            idx1 = i

            y = self.data_samples[idx1][1]
            # to avoid idx1 = idx2, if idx1 > half, choose idx2 randomly from bottom half and the opposite
            # to avoid two digits of the same label being joined
            labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            y2 = y
            while y == y2:
                if i > int(round(self.number_samples / 2)):
                    idx2 = random.randint(0, int(round(self.number_samples / 2)))
                else:
                    idx2 = random.randint(int(round(self.number_samples / 2)), self.number_samples - 1)

                y2 = self.data_samples[idx2][1]

            image1 = np.array(self.data_samples[idx1][0])
            image2 = np.array(self.data_samples[idx2][0])

            # digit1, digit2 = self.resize_images(digit1, digit2)
            # Select as negative label a label not in any of the mixed images
            labels.remove(y)
            labels.remove(y2)
            y_neg = labels[random.randint(0, 7)]

            rand = random.randint(0,3)

            # if rand == 6:
            #     rand = random.randint(0, 2)
            #     hybrid1[:, :, rand] = image1[:, :, rand]
            # # Create hybrid images by adding together two images
            if rand > 1:
                w1 = random.uniform(0.05, 0.2)
                w2 = 1-w1
                hybrid1 = cv2.addWeighted(image1, w1, image2, w2, 0)
            else:
                hybrid1 = image2.copy()

            if rand < 2:
                w1 = random.uniform(0.6, 0.95)
                w2 = 1-w1
                image1 = cv2.addWeighted(image1, w1, image2, w2, 0)

            # self.visualize(image1, image2, hybrid1)
            # print(y)

            hybrid_imgs.append(hybrid1)
            y_negs.append(y_neg)
            normal_imgs.append(image1)
            y_pos.append(y)

        return normal_imgs, hybrid_imgs, y_pos, y_negs

    def __getitem__(self, index):

        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        # Load Positive and Negative Samples
        x_pos = transform(self.normal_imgs[index])
        y_pos = self.y_pos[index]
        x_neg = transform(self.hybrid_imgs[index])
        y_neg = self.y_neg[index]

        # Transform Positive and Negative Samples
        x_pos = x_pos.reshape(3, self.size[1], self.size[0])  #
        y_pos = torch.from_numpy(np.asarray(y_pos))

        x_neg = x_neg.reshape(3, self.size[1], self.size[0])  #
        y_neg = torch.from_numpy(np.asarray(y_neg))

        return (x_pos, y_pos), (x_neg, y_neg)

    #
    # def __getitem__(self, index):
    #
    #     transform = Compose([
    #         ToTensor(),
    #         Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])])
    #
    #     # Load Positive and Negative Samples
    #     x_pos = transform(self.normal_imgs[index])
    #     y_pos = self.y_pos[index]
    #     x_neg = transform(self.hybrid_imgs[index])
    #     y_neg = self.y_neg[index]
    #
    #     # Transform Positive and Negative Samples
    #     x_pos = x_pos.reshape(3, self.size[1], self.size[0])  #
    #     y_pos = torch.from_numpy(np.asarray(y_pos))
    #
    #     x_neg = x_neg.reshape(3, self.size[1], self.size[0])  #
    #     y_neg = torch.from_numpy(np.asarray(y_neg))
    #
    #     return (x_pos, y_pos), (x_neg, y_neg)


    def __len__(self):
        return len(self.normal_imgs)


class Hybrid_CIFAR10_test(torch.utils.data.Dataset):
    def __init__(self, data_samples, size=(32, 32), number_samples=60000, increment_dif=False):

        self.data_samples = data_samples
        self.size = size
        self.repeats = 20
        self.threshold = 0.2
        self.threshold_decay = 0.95
        self.number_samples = number_samples  # how train data to create many to create

        self.normal_imgs, self.y_pos = self.generate_data()

    def resize_images(self, digit1, digit2):

        # Resize images to the same size
        digit1 = cv2.resize(digit1, self.size)
        digit2 = cv2.resize(digit2, self.size)

        return digit1, digit2

    def visualize(self, img1, img2, hybrid):

        # Plot the original images, the binary mask, and the hybrid images
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 5))
        ax1.imshow(img1)
        ax1.set_title('Image 1')
        ax2.imshow(img2)
        ax2.set_title('Image 2')
        ax3.imshow(hybrid)
        ax3.set_title('Hybrid')
        plt.show()
        return

    def generate_data(self):

        normal_imgs = []
        y_pos = []

        for i in range(self.number_samples):
            idx1 = i
            y = self.data_samples[idx1][1]
            image1 = np.array(self.data_samples[idx1][0])

            # self.visualize(image1, image2, hybrid1)
            # print(y)

            normal_imgs.append(image1)
            y_pos.append(y)

        return normal_imgs,  y_pos

    def __getitem__(self, index):

        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        # Load Positive and Negative Samples
        x_pos = transform(self.normal_imgs[index])
        y_pos = self.y_pos[index]

        # Transform Positive and Negative Samples
        x_pos = x_pos.reshape(3, self.size[1], self.size[0])  #
        y_pos = torch.from_numpy(np.asarray(y_pos))

        return x_pos, y_pos


    def __len__(self):
        return len(self.normal_imgs)


class HybridImages_Train(torch.utils.data.Dataset):
    def __init__(self, data_samples, size=(28, 28), number_samples=60000, dataset='MNIST', increment_dif=False):

        self.data_samples = data_samples
        self.size = size
        self.repeats = 20
        self.threshold = 0.2
        self.threshold_decay = 0.95
        self.number_samples = number_samples  # how train data to create many to create
        self.increment_dif = increment_dif
        self.dataset = dataset
        # Define a custom filter
        filter_kernel1 = np.array([
            [0, 1, 2, 1, 0],
            [1, 2, 4, 2, 1],
            [2, 4, 10, 4, 2],
            [1, 2, 4, 2, 1],
            [0, 1, 2, 1, 0]], dtype=np.float32) / 50  # Normalize the kernel

        filter_kernel2 = np.array([
            [1, 2, 1]
        ], dtype=np.float32) / 4  # Normalize the kernel

        self.filters = [filter_kernel1, filter_kernel2]

        # create random masks
        self.masks = self.random_regions_mask()
        # generate data
        self.normal_imgs, self.hybrid_imgs, self.y_pos, self.y_neg = self.generate_data()

    def square_mask(self):
        # # Create a binary mask with large regions of ones and zeros
        mask = np.zeros(self.size, dtype=np.uint8)
        sq_bot = np.random.randint(15, int(round(self.size[0] / 2)))
        sq_top = np.random.randint(sq_bot + 15, self.size[0] - 1)
        mask[sq_bot:sq_top, sq_bot:sq_top] = 1  # Set a square region of ones in the center of the image

        return mask

    def random_regions_mask(self, threshold=False):
        # Generate a random bit image
        masks = []
        for j in range(self.number_samples):
            smoothed = np.random.randint(0, 2, (self.size[1], self.size[0]), dtype=np.uint8)
            for j in range(self.repeats):
                # Apply the filter to the image using convolution
                smoothed = cv2.filter2D(smoothed, -1, self.filters[0])
                # print(smoothed[18:25, 18:25])
                # threshold image
                if threshold:
                    _, smoothed = cv2.threshold(smoothed, self.threshold, 1, cv2.THRESH_BINARY)

            masks.append(smoothed)

        return masks

    def resize_images(self, digit1, digit2):

        # Resize images to the same size
        digit1 = cv2.resize(digit1, self.size)
        digit2 = cv2.resize(digit2, self.size)

        return digit1, digit2

    def thresh_decay(self):

        self.threshold *= self.threshold_decay
        # print(threshold)

        return

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

        hybrid_imgs = []
        y_negs = []
        normal_imgs = []
        y_pos = []

        for i in range(self.number_samples):
            labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            idx1 = i

            y = self.data_samples[idx1][1]
            # to avoid idx1 = idx2, if idx1 > half, choose idx2 randomly from bottom half and the opposite
            # to avoid two digits of the same label being joined

            y2 = y
            while y == y2:
                if i > int(round(self.number_samples / 2)):
                    idx2 = random.randint(0, int(round(self.number_samples / 2)))
                else:
                    idx2 = random.randint(int(round(self.number_samples / 2)), self.number_samples - 1)

                y2 = self.data_samples[idx2][1]

            digit1 = np.array(self.data_samples[idx1][0])
            digit2 = np.array(self.data_samples[idx2][0])

            labels.remove(y)
            labels.remove(y2)
            y_neg = labels[random.randint(0, 7)]

            # digit1, digit2 = self.resize_images(digit1, digit2)

            # select mask
            mask = self.masks[i]

            masked_img1 = cv2.bitwise_and(digit1, digit1, mask=mask)
            masked_img2 = cv2.bitwise_and(digit2, digit2, mask=1 - mask)

            # Create hybrid images by adding together one image times the mask and a different image times the reverse of the mask
            hybrid1 = cv2.addWeighted(masked_img2, 1, masked_img1, 1, 0)
            if self.increment_dif:
                ix = random.randint(0, self.number_samples - 1)
                mask = self.masks[ix]
                masked_img1 = cv2.bitwise_and(hybrid1, hybrid1, mask=mask)
                masked_img2 = cv2.bitwise_and(digit1, digit1, mask=1 - mask)

                # Create hybrid images by adding together one image times the mask and a different image times the reverse of the mask
                hybrid1 = cv2.addWeighted(masked_img2, 1, masked_img1, 1, 0)

            # self.visualize(digit1, digit2, mask, hybrid1)
            # print(y)


            hybrid_imgs.append(hybrid1)
            y_negs.append(y_neg)
            normal_imgs.append(digit1)
            y_pos.append(y)

        return normal_imgs, hybrid_imgs, y_pos, y_negs

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
        x_neg = transform(self.hybrid_imgs[index])
        y_neg = self.y_neg[index]

        # img = cv2.resize(img, (self.img_w, self.img_h),
        #                  interpolation=cv2.INTER_NEAREST)

        # Transform Positive and Negative Samples
        x_pos = x_pos.reshape(1, self.size[1], self.size[0])  #
        y_pos = torch.from_numpy(np.asarray(y_pos))

        x_neg = x_neg.reshape(1, self.size[1], self.size[0])  #
        y_neg = torch.from_numpy(np.asarray(y_neg))

        return (x_pos, y_pos), (x_neg, y_neg)

    def __len__(self):
        return len(self.normal_imgs)


class HybridImages_Test(torch.utils.data.Dataset):
    def __init__(self, data_samples, size=(28, 28), number_samples=60000, dataset='MNIST',  increment_dif=False):

        self.data_samples = data_samples
        self.size = size
        self.repeats = 20
        self.threshold = 0.2
        self.threshold_decay = 0.95
        self.number_samples = number_samples  # how train data to create many to create
        self.increment_dif = increment_dif
        self.dataset = dataset
        self.normal_imgs, self.y_pos = self.generate_data()


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
            # to avoid idx1 = idx2, if idx1 > half, choose idx2 randomly from bottom half and the opposite
            # to avoid two digits of the same label being joined


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

        # img = cv2.resize(img, (self.img_w, self.img_h),
        #                  interpolation=cv2.INTER_NEAREST)

        # Transform Positive and Negative Samples
        x_pos = x_pos.reshape(1, self.size[1], self.size[0])  #
        y_pos = torch.from_numpy(np.asarray(y_pos))

        return x_pos, y_pos

    def __len__(self):
        return len(self.normal_imgs)