import random
import torch
import torch.nn as nn
import numpy as np
import csv
import os


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    batch_range = range(x.shape[0])
    x_[batch_range, y] = x.max()

    return x_


def overlay_y_on_x3d(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """

    B, C, H, W = x.shape
    unflatten = nn.Unflatten(1, torch.Size([C, H, W]))

    x_ = x.clone()
    x_ = x_.reshape(x_.size(0), -1)

    x_[:, :10] *= 0.0
    x_[range(x_.shape[0]), y] = x_.max()

    x_ = unflatten(x_)


    return x_


def overlay_y_on_x4d(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """

    B, C, H, W = x.shape

    unflatten = nn.Unflatten(1, torch.Size([H, W]))

    x_ = x.clone()
    for ch in range(C):
        channel = x_[:, ch]
        # print(channel.shape)
        channel = channel.reshape(channel.size(0), -1)
        channel[:, :10] *= 0.0
        channel[range(channel.shape[0]), y] = channel.max()
        channel = unflatten(channel)
        # print(channel.shape)
        x_[:, ch, :, :] = channel

    return x_


def channel_shuffle(y, groups=2):
    # Split activations by Number_of_classes sets   `
    y_sets = torch.split(y, groups, dim=1)

    setrange = list(range(0, len(y_sets)))
    # print(setrange)

    for i in range(len(y_sets)):
        rand = random.randint(0, len(setrange)-1)
        yset = y_sets[setrange[rand]]
        setrange.pop(rand)

        if i == 0:
            group_y = yset
        else:
            group_y = torch.cat((group_y, yset), 1)

        # print(group_y.shape)

    return group_y


def save_model(model, model_id, dataset, epoch):
    b1_statedict = []
    b2_statedict = []
    nn_statedict = []
    b1_opt_statedict = []
    b2_opt_statedict = []
    nn_opt_statedict = []

    for i, layer in enumerate(model.convb1_layers):
        b1_statedict.append(layer.state_dict())
        b1_opt_statedict.append(layer.opt.state_dict())

    for i, layer in enumerate(model.convb2_layers):
        b2_statedict.append(layer.state_dict())
        b2_opt_statedict.append(layer.opt.state_dict())

    for i, layer in enumerate(model.nn_layers):
        nn_statedict.append(layer.state_dict())
        nn_opt_statedict.append(layer.opt.state_dict())

    checkpoint = {'model': model,
                  'SF_state_dict': model.classifier_b1.state_dict(),
                  'SF_optimizer': model.classifier_b1.opt.state_dict(),
                  'b1_state_dict': b1_statedict,
                  'b1_opt_state_dict': b1_opt_statedict,
                  'b2_state_dict': b2_statedict,
                  'b2_opt_state_dict': b2_opt_statedict,
                  'nn_state_dict': nn_statedict,
                  'nn_opt_state_dict': nn_opt_statedict
                  }

    torch.save(checkpoint, './weights/' + dataset + '/' + model_id + str(epoch) + '.pth')


def save_state(model, model_id, dataset, epoch):
    statedict = []
    nn_statedict = []
    opt_statedict = []
    nn_opt_statedict = []

    for i, layer in enumerate(model.conv_layers):
        statedict.append(layer.state_dict())
        opt_statedict.append(layer.opt.state_dict())

    for i, layer in enumerate(model.nn_layers):
        nn_statedict.append(layer.state_dict())
        nn_opt_statedict.append(layer.opt.state_dict())

    checkpoint = {'model': model,
                  'SF_state_dict': model.classifier_b1.state_dict(),
                  'SF_optimizer': model.classifier_b1.opt.state_dict(),
                  'state_dict': statedict,
                  'opt_state_dict': opt_statedict,
                  'nn_state_dict': nn_statedict,
                  'nn_opt_state_dict': nn_opt_statedict
                  }

    torch.save(checkpoint, './weights/' + dataset + '/' + model_id + str(epoch) + '.pth')


def save_FFrep(model, model_id, dataset, epoch):

    nn_statedict = []
    nn_opt_statedict = []
    for i, layer in enumerate(model.layers):
        nn_statedict.append(layer.state_dict())
        nn_opt_statedict.append(layer.opt.state_dict())

    checkpoint = {'model': model,
                  'nn_state_dict': nn_statedict,
                  'nn_opt_state_dict': nn_opt_statedict
                  }

    torch.save(checkpoint, './weights/' + dataset + '/' + model_id + str(epoch) + '.pth')


def load_model(model, model_id, dataset, epoch, param=False):
    checkpoint = torch.load('./weights/' + dataset + '/' + model_id + str(epoch) + '.pth')

    for i, layer in enumerate(model.conv_layers):
        layer.load_state_dict(checkpoint['state_dict'][i])
        layer.opt.load_state_dict(checkpoint['opt_state_dict'][i])

    for i, layer in enumerate(model.nn_layers):
        layer.load_state_dict(checkpoint['nn_state_dict'][i])
        layer.opt.load_state_dict(checkpoint['nn_opt_state_dict'][i])

    model.classifier_b1.load_state_dict(checkpoint['SF_state_dict'])
    model.classifier_b1.opt.load_state_dict(checkpoint['SF_optimizer'])

    for parameter in model.parameters():
        parameter.requires_grad = param

    return model


def save_traininglog(loss_log,filename, layer_losses=True):

    # Specify the output CSV file path
    csv_file_path = './TrRes/' + filename + '.csv'

    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the column headers
        if layer_losses:
            # writer.writerow(['B1_Conv', 'B1_G_Conv', 'B2_Conv', 'B2_G_Conv', 'Gd_NN_1', 'Gd_NN_2', 'Sf_NN'])
            writer.writerow(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'])
        else:
            loss_log = np.transpose(loss_log)
            writer.writerow(['Avg_Train', 'Avg_Test', 'SF_Train', 'SF_Test', 'Gd_Train', 'Gd_Test'])

        # Write the data rows
        for i in range(np.shape(loss_log)[0]):
            epoch_losses = np.asarray(loss_log)[i, :]
            writer.writerow(epoch_losses)


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.initial_seed(),
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


