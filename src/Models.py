
from Layer_cnn import *

#  PAPER RUNS

class CW_Comp(torch.nn.Module):

    def __init__(self, out_channels_list, batch_size, CFSE=False, sf_pred=False, dataset='MNIST', ILT='Acc', loss_='CwC', N_Classes=10):
        super(CW_Comp, self).__init__()
        self.iter = 1

        self.batch_size = batch_size
        self.show_iters = 800
        self.sf_pred = sf_pred
        self.nn_layers = []
        self.conv_layers = nn.ModuleList()

        if dataset == 'MNIST':
            if ILT == 'Fast':
                self.start_end = [[0, 3], [1, 4], [2, 5], [3, 6], [4, 20], [5, 20]]
            else:
                # self.start_end = [[0, 6], [0, 11], [0, 16], [0, 21], [0, 20], [0, 20]]
                self.start_end = [[0, 2], [0, 3], [0, 4], [0, 5], [0, 20], [0, 20]]
            CNN_l1_dims = [1, 28, 28]
        elif dataset == 'FMNIST':
            if ILT == 'Fast':
                self.start_end = [[0, 7], [1, 10], [2, 13], [3, 16], [4, 30], [5, 40]]
            else:
                self.start_end = [[0, 10], [0, 15], [0, 19], [0, 23], [0, 36], [0, 50]]
                # self.start_end = [[0, 6], [0, 9], [0, 11], [0, 14], [0, 30], [0, 40]]
            CNN_l1_dims = [1, 28, 28]
        else:
            if ILT == 'Fast':
                self.start_end = [[0, 11], [2, 18], [4, 26], [6, 32], [8, 36], [10, 50]]
            else:
                self.start_end = [[0, 11], [0, 16], [0, 21], [0, 25], [0, 36], [0, 50]]
            CNN_l1_dims = [3, 32, 32]

        # # # CNN LAYERS # # #
        kernel_size = 3

        self.n_classes = N_Classes
        # Dynamically add layers
        dims = [CNN_l1_dims]

        self.final_channels = out_channels_list[-1]

        for i, out_channels in enumerate(out_channels_list):
            if i % 2 == 1 and CFSE:
                group = self.n_classes
            else:
                group = 1
            in_channels = dims[-1][0]
            layer = Conv_Layer(dims[-1], in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, maxpool=(i % 2 == 1), groups=group, droprate=0, loss_criterion=loss_).cuda()
            self.conv_layers.append(layer)
            dims.append(layer.next_dims)

        # # # # Softmax Classifiers
        self.classifier_b1 = Softmax_CF(self.conv_layers[-1].N_neurons_out).cuda()

    def predict(self, x):
        goodness_per_label = []
        h = x
        for i, convlayer in enumerate(self.conv_layers):
            h = convlayer.ff_infer(h)  # The hidden activities in all but not the CNN layers are used for Prediction

        # Step 1: Reshape tensor to [B, C, NChannels/C, H, W]
        h_reshaped = h.view(h.shape[0], self.n_classes, self.final_channels // self.n_classes, h.shape[2], h.shape[3])

        # Step 2: Compute mean squared value for each subset
        mean_squared_values = (h_reshaped ** 2).mean(dim=[2, 3, 4])

        # Step 3: Identify subset with max mean squared value
        _, predicted_classes = torch.max(mean_squared_values, dim=1)

        if self.sf_pred:
            sf = self.classifier_b1(h)
            return predicted_classes, sf.argmax(1)
        else:
            return predicted_classes

    def train_(self, x_pos, y, y_n, epoch):

        start_end = self.start_end

        if self.iter % self.show_iters == 0:
            show = True
        else:
            show = False

        h_pos = x_pos

        for i, convlayer in enumerate(self.conv_layers):
            s, e = start_end[i]
            if epoch in list(range(s, e)):
                h_pos = convlayer.forward_forward(h_pos, y.cuda(), show)
                if show:
                    print('Training Conv Layer: ', i, '...')
            else:
                h_pos = convlayer.ff_infer(h_pos)

        b1_out = h_pos.clone()  # Block 1 out

        h_pos = overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
        h_neg = overlay_y_on_x3d(h_pos, y_n)

        for i, nnlayer in enumerate(self.nn_layers):

            s, e = start_end[i + len(self.conv_layers) + len(self.convb2_layers)]
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
            self.classifier_b1.train_classifier(b1_out, y.cuda(), show)

        self.iter += 1


class CW_Comp_ClassGroup(torch.nn.Module):

    def __init__(self, out_channels_list, n_classes, batch_size, CFSE=False, sf_pred=False, dataset='MNIST', ILT='Acc', loss_='CwC'):
        super(CW_Comp_ClassGroup, self).__init__()
        self.iter = 1

        self.batch_size = batch_size
        self.show_iters = 800
        self.sf_pred = sf_pred
        self.nn_layers = []
        self.conv_layers = nn.ModuleList()

        if ILT == 'Fast':
            self.start_end = [[0, 11], [2, 18], [4, 26], [6, 32], [8, 36], [10, 50]]
        else:
            self.start_end = [[0, 30], [0, 30], [30, 60], [30, 60], [60, 85], [60, 85], [85, 100], [85, 100]]
            # self.start_end = [[0, 1], [0, 1], [1, 2], [1, 2], [2, 3], [2, 3], [3, 4], [3, 4]]
            CNN_l1_dims = [3, 32, 32]

        # # # CNN LAYERS # # #
        kernel_size = 3
        self.n_classes = n_classes
        # Dynamically add layers
        dims = [CNN_l1_dims]

        self.final_channels = out_channels_list[-1]

        for i, out_channels in enumerate(out_channels_list):
            if i % 2 == 1 and CFSE:
                group = self.n_classes[i]
            else:
                group = 1

            if max(n_classes) == n_classes[i]:
                class_groups = None
            else:
                class_groups = int(max(n_classes)/n_classes[i])

            in_channels = dims[-1][0]
            layer = Conv_Layer(dims[-1], in_channels=in_channels, out_channels=out_channels, num_classes=self.n_classes[i],
                              kernel_size=kernel_size, maxpool=(i % 2 == 1), groups=group, droprate=0, loss_criterion=loss_, ClassGroups=class_groups).cuda()
            self.conv_layers.append(layer)
            dims.append(layer.next_dims)

        # # # # Softmax Classifiers
        self.classifier_b1 = Softmax_CF(self.conv_layers[-1].N_neurons_out, self.n_classes[-1]).cuda()

    def predict(self, x):
        goodness_per_label = []
        h = x
        for i, convlayer in enumerate(self.conv_layers):
            h = convlayer.ff_infer(h)

        # Step 1: Reshape tensor to [B, C, NChannels/C, H, W]
        h_reshaped = h.view(h.shape[0], self.n_classes[-1], self.final_channels // self.n_classes[-1], h.shape[2], h.shape[3])

        # Step 2: Compute mean squared value for each subset
        mean_squared_values = (h_reshaped ** 2).mean(dim=[2, 3, 4])

        # Step 3: Identify subset with max mean squared value
        _, predicted_classes = torch.max(mean_squared_values, dim=1)

        if self.sf_pred:
            sf = self.classifier_b1(h)
            return predicted_classes, sf.argmax(1)
        else:
            return predicted_classes

    def train_(self, x_pos, y, y_n, epoch):

        start_end = self.start_end

        if self.iter % self.show_iters == 0:
            show = True
        else:
            show = False

        h_pos = x_pos

        for i, convlayer in enumerate(self.conv_layers):
            s, e = start_end[i]
            if epoch in list(range(s, e)):
                h_pos = convlayer.forward_forward(h_pos, y.cuda(), show)
                if show:
                    print('Training Conv Layer: ', i, '...')
            else:
                h_pos = convlayer.ff_infer(h_pos)

        b1_out = h_pos.clone()  # Block 1 out

        h_pos = overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
        h_neg = overlay_y_on_x3d(h_pos, y_n)

        for i, nnlayer in enumerate(self.nn_layers):

            s, e = start_end[i + len(self.conv_layers) + len(self.convb2_layers)]
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
            self.classifier_b1.train_classifier(b1_out, y.cuda(), show)

        self.iter += 1
