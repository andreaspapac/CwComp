from Layer_fc import *


class Conv_Layer(nn.Conv2d):
    def __init__(self, in_dims, in_channels=1, out_channels=8, num_classes=10, kernel_size=7, stride=1, padding=1,
                 maxpool=True, groups=1, droprate=0, loss_criterion='CwC_CE', ClassGroups=None):
        super(Conv_Layer, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, groups=groups)

        self.dims = in_dims
        self.outc = out_channels
        self.kernel_size = kernel_size
        self.outsize = int(((self.dims[1] - self.kernel_size + (2 * padding)) / stride) + 1)
        self.ismaxpool = maxpool
        self.groups = groups
        self.loss = 10000
        self.ClassGroups = ClassGroups

        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.outsize = int(self.outsize / 2)  # if maxpool

        self.wsz = self.outsize  # for visualisation of features
        self.N_neurons_out = self.outc * self.outsize ** 2  # total number of output neurons
        self.next_dims = [self.outc, self.outsize, self.outsize]

        self.conv_bn = nn.BatchNorm2d(self.outc, eps=1e-4)
        self.dropout = torch.nn.Dropout(p=droprate)
        self.lr = 0.01  # 0.01
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
        self.gf = None

        print(self.dims, self.N_neurons_out)

    def forward(self, x):

        x = self.dropout(x)

        # Forward Pass
        x = self._conv_forward(x, self.weight, self.bias)
        x = F.relu(x, inplace=True)
        if self.ismaxpool:
            x = self.maxpool(x)
        y = self.conv_bn(x)
        # y = F.batch_norm(x, self.conv_bn.running_mean, self.conv_bn.running_var, self.conv_bn.weight, self.conv_bn.bias,
        #                  self.conv_bn.training, self.conv_bn.momentum, self.conv_bn.eps)

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
            data_min = features.min()
            data_max = features.max()
            for cb, channelblock in enumerate(feature_sets):
                cb_rep = channelblock.max(0).values
                axs[cb].imshow(cb_rep.detach().cpu(), cmap='jet', vmin=data_min, vmax=data_max)

            plt.show()

    def goodness_factorCW(self, y, gt):
        # y : R^BxCxHxW

        # Vectorized pos_mask creation
        pos_mask = torch.zeros((gt.shape[0], self.num_classes), dtype=torch.uint8, device=y.device)
        arange_tensor = torch.arange(gt.shape[0], device=y.device)
        pos_mask[arange_tensor, gt] = 1
        pos_mask = pos_mask.bool()
        neg_mask = ~pos_mask

        # Split activations by Number_of_classes sets
        y_sets = torch.split(y, int(self.outc / self.num_classes), dim=1)

        goodness_factors = [y_set.pow(2).mean((1, 2, 3)).unsqueeze(-1) for y_set in y_sets]
        gf = torch.cat(goodness_factors, 1)

        # Efficiently compute g_pos and g_neg without reshaping
        g_pos = gf[pos_mask].view(-1, 1)
        g_neg = gf[neg_mask].view(gf.shape[0], -1).mean(1).unsqueeze(-1)

        return g_pos, g_neg, gf

    def epoch_loss(self):
        epl_mean = torch.tensor(self.ep_losses).mean().item()
        # if abs(epl_mean - self.ep_losses[-1]) < 0.00001:
        #     self.lr_decay()
        #     print('lr decay, new lr = ', self.lr)
        self.ep_losses = []
        # print(self.ep_losses)
        # print('ep losses', self.ep_losses)
        return epl_mean

    def forward_forward(self, x_pos, gt, show):

        # forward pass
        y_pos = self.forward(x_pos)

        if self.ClassGroups is not None:
            # convert classes into superclasses
            gt = gt // self.ClassGroups

        gt = gt.cuda()

        g_pos, g_neg, self.gf = self.goodness_factorCW(y_pos, gt)

        if self.loss_criterion == 'CwC':
            loss = self.criterion(g_pos, self.gf)
        elif self.loss_criterion == 'CwC_CE':
            loss = self.criterion(self.gf, gt)
        elif self.loss_criterion == 'PvN':
            loss = self.criterion(g_pos, g_neg)
        elif self.loss_criterion == 'CWG':
            loss = self.criterion(g_pos, g_neg, self.gf, gt)

        self.loss = loss
        # print(loss)
        self.ep_losses.append(loss)

        if show:
            print('------------------------------')
            print('g_pos_CW: {}, g_neg_CW:{}'.format(g_pos.mean().item(), g_neg.mean().item()))
            print('Loss: {}'.format(loss))

        self.opt.zero_grad()
        # this backward just compute the derivative and hence
        # is not considered backpropagation.
        loss.backward()
        self.opt.step()

        return y_pos.detach()

    def ff_infer(self, x):

        # Forward Pass
        x = self._conv_forward(x, self.weight, self.bias)
        x = F.relu(x, inplace=True)
        if self.ismaxpool:
            x = self.maxpool(x)
        y = self.conv_bn(x)
        # y = F.batch_norm(x, self.conv_bn.running_mean, self.conv_bn.running_var, self.conv_bn.weight, self.conv_bn.bias,
        #                  self.conv_bn.training, self.conv_bn.momentum, self.conv_bn.eps)
        return y.detach()


class CwCLoss(nn.Module):
    def __init__(self):
        super(CwCLoss, self).__init__()
        self.eps = 1e-9

    def forward(self, g_pos, logits):
        # Ensure that values are not too large/small for exp function
        logits = torch.clamp(logits, min=-50, max=50)
        g_pos = torch.clamp(g_pos, min=-50, max=50)

        # Calculate the sum of the exponential of all goodness scores for each sample
        exp_sum = torch.sum(torch.exp(logits), dim=1)

        # Compute the CwC loss using the formula
        loss = -torch.mean(torch.log((torch.exp(g_pos) + self.eps) / (exp_sum + self.eps)))

        return loss


class PvNLoss(nn.Module):
    def __init__(self):
        super(PvNLoss, self).__init__()
        self.threshold = 2.0

    def forward(self, g_pos, g_neg):
        # error is if goodness for positive data is low -(g_pos - threshold)
        # or if goodness for negative data is high +(gpos - threshold)
        errors = torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold])

        loss = torch.log(1 + torch.exp(errors)).mean()

        return loss


class CWGLoss(nn.Module):
    def __init__(self, w1):
        super(CWGLoss, self).__init__()
        self.w1 = w1
        self.loss1 = nn.CrossEntropyLoss()  # CwCLoss()
        self.loss2 = PvNLoss()

    def forward(self, g_pos, g_neg, GF, gt):
        loss1 = self.loss1(GF, gt)
        loss2 = self.loss2(g_pos, g_neg)
        loss = (self.w1 * loss1 + (1 - self.w1) * loss2)

        return loss


