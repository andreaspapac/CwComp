from Datasets import *
from torch.utils.data import DataLoader
from utils import *
import numpy as np
from torch.autograd import Variable
from Models import *
import os

CUDA_LAUNCH_BLOCKING = 1.

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.initial_seed(),
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

Avg_train_losses = []
Avg_test_losses = []
Sf_train_losses = []
Sf_test_losses = []
layerwise_loss = []


data_path = 'C:/Users/Andreas/Desktop/PhD/NeurIPS_Re/src/data/'
seed = 2          # 52/13/22/2

loss_criterion = 'CwC_CE'  # 'CwC', 'CwC_CE', 'PvN', 'CWG'
dataset = 'MNIST'  # MNIST/FMNIST/CIFAR
ILT = 'Acc'
save_ = True
CFSE = True
sf_pred = True
ClassGroup = False
retrain = False
lr_decay = False

stage = 0
batch_size = 128
num_workers = 4
channels_list = [20, 80, 240, 480]  # [40, 80, 160, 320] #[20, 80, 240, 480]
show_iters = 200
n_epochs_d = 90
N_Classes = 10
min_testerror = 0.3

# # Evaluation # #
epoch = 1
visualize_features = False

if CFSE:
    architecture = 'CFSE'
else:
    architecture = 'FFCNN'
if sf_pred:
    preds = 'AvgSf'
else:
    preds = 'Avg'

if dataset[:5] == 'CIFAR':
    if dataset == 'CIFAR':
        print('CIFAR-10')
        train_data = CIFAR10(data_path, train=True, download=True)
        test_data = CIFAR10(data_path, train=False, download=True)
    else:
        print('CIFAR-100')
        train_data = CIFAR100(data_path, train=True, download=True)
        test_data = CIFAR100(data_path, train=False, download=True)
        ClassGroup = True
        channels_list = [60, 120, 240, 400, 800, 1600]
        N_Classes = [20, 20, 20, 20, 100, 100]

    train_dataset = X_CIFAR(train_data, number_samples=50000)
    test_dataset = X_CIFAR(test_data, number_samples=10000)
    sf_min_testerror = 0.8
else:
    if dataset == 'MNIST':
        train_data = MNIST(data_path, train=True, download=True)
        test_data = MNIST(data_path, train=False, download=True)
    elif dataset == 'FMNIST':
        train_data = FashionMNIST(data_path, train=True, download=True)
        test_data = FashionMNIST(data_path, train=False, download=True)
    train_dataset = X_MNIST(train_data, number_samples=60000, dataset=dataset)
    test_dataset = X_MNIST(test_data, number_samples=10000, dataset=dataset)

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model_id = architecture + '_' + loss_criterion + '_' + preds + '_' + dataset + '_' + ILT + '_batch' + str(batch_size) + '_Ch' + 'n'.join(map(str, channels_list)) + '_stage' + str(stage)


if __name__ == "__main__":
    # torch.manual_seed(1234)
    seed_everything(seed=52)

    # LOAD MODEL
    if ClassGroup:
        model = CW_Comp_ClassGroup(channels_list, N_Classes, batch_size=batch_size, CFSE=CFSE, sf_pred=sf_pred, dataset=dataset, ILT=ILT, loss_=loss_criterion)
    else:
        model = CW_Comp(channels_list, batch_size=batch_size, CFSE=CFSE, sf_pred=sf_pred, dataset=dataset, ILT=ILT, loss_=loss_criterion, N_Classes=N_Classes)
    model = load_model(model, model_id, dataset, epoch)

    epoch_errors = []
    sf_epoch_errors = []

    model.eval()
    with torch.no_grad():
        for step, (x_p, y_p) in enumerate(test_loader):
            x_p = Variable(x_p).cuda()
            y_p = y_p.long()

            if sf_pred:
                pred, sf = model.predict(x_p)
                sf_batch_loss = 1.0 - sf.eq(y_p.cuda()).float().mean().item()
                sf_epoch_errors.append(sf_batch_loss)
            else:
                pred = model.predict(x_p)

            if visualize_features:
                print('Class: {}'.format(y_p[0]))
                for i, layer in enumerate(model.conv_layers):
                    if i == 0:
                        y_pos = layer.forward(x_p).detach()
                    else:
                        y_pos = layer.forward(y_pos).detach()
                    print(i)
                    layer.vis_features(y_pos[0])

            batch_loss = 1.0 - pred.eq(y_p.cuda()).float().mean().item()
            epoch_errors.append(batch_loss)

    # # PRINTS FOR EVERY EPOCH
    print('Epoch: {}'.format(epoch))
    # print('Avg Pred - Train Error = {}'.format(np.asarray(epoch_losses).mean()))
    print('Avg Pred - Test Error = {}'.format(np.asarray(epoch_errors).mean()))
    if sf_pred:
        # print('Sf Pred -- Train Error = {}'.format(np.asarray(sf_epoch_losses).mean()))
        print('Sf Pred -- Test Error = {}'.format(np.asarray(sf_epoch_errors).mean()))

