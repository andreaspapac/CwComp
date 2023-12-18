import argparse
from Datasets import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Configure the hyperparameters")

    parser.add_argument("--data_path", default='C:/Users/Andreas/Desktop/PhD/NeurIPS_Re/src/data/', type=str,
                        help="Data Path")
    parser.add_argument("--seed", default=2, type=int, help="Torch Random Seed")   # 52/13/22/2
    parser.add_argument("--loss_criterion", default='CwC_CE', type=str,
                        help="Loss function: {'CwC', 'CwC_CE', 'PvN', 'CWG'}")
    parser.add_argument("--dataset", default='CIFAR', type=str,
                        help="Dataset: {MNIST, FMNIST, CIFAR, CIFAR100}")
    parser.add_argument("--ILT", default='Acc', type=str,
                        help="ILT Strategy: {Acc, Fast}")
    parser.add_argument("--save", default='True', type=str,
                        help="Save Weights")
    parser.add_argument("--CFSE", default='True', type=str,
                        help="CFSE Architecture: {True=CFSE, False=FFCNN}")
    parser.add_argument("--sf_pred", default='True', type=str,
                        help="Enable Sf Predictor: {True=Avg+Sf, False=Sf}")
    parser.add_argument("--ClassGroup", default='False', type=str,
                        help="Class Grouping - Used for CIFAR100")
    parser.add_argument("--retrain", default='False', type=str,
                        help="Retrain Model: {True=Load Weights, False=from Scratch}")
    parser.add_argument("--stage", default=0, type=int,
                        help="Training Stage: {0 = from scratch}")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch Size")
    # parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--channels_list", default=[20, 80, 240, 480], type=list,
                        help="Number of Channels per Layer")  # [40, 80, 160, 320] #[20, 80, 240, 480]
    parser.add_argument("--show_iters", default=200, type=int,
                        help="Iteration frequency for Visualizing Layerwise Stats")
    parser.add_argument("--n_epochs", default=20, type=int,
                        help="Total Number of Training Epochs")
    parser.add_argument("--N_Classes", default=10, type=int,
                        help="Number of Classes")
    parser.add_argument("--min_testerror", default=0.32, type=float,
                        help="Minimum Testing Error for Weight Saving")
    parser.add_argument("--load_epoch", default=5, type=int,
                        help="Model Weights Epoch")

    args = parser.parse_args()

    args.CFSE = str2bool(args.CFSE)
    args.save = str2bool(args.save)
    args.retrain = str2bool(args.retrain)
    args.sf_pred = str2bool(args.sf_pred)
    args.ClassGroup = str2bool(args.ClassGroup)

    return args


def configure(args):

    if args.CFSE:
        architecture = 'CFSE'
    else:
        architecture = 'FFCNN'
    if args.sf_pred:
        preds = 'AvgSf'
    else:
        preds = 'Avg'

    if args.dataset[:5] == 'CIFAR':
        if args.dataset == 'CIFAR':
            print('CIFAR-10')
            train_data = CIFAR10(args.data_path, train=True, download=True)
            test_data = CIFAR10(args.data_path, train=False, download=True)
        else:
            print('CIFAR-100')
            train_data = CIFAR100(args.data_path, train=True, download=True)
            test_data = CIFAR100(args.data_path, train=False, download=True)
            args.ClassGroup = True
            args.channels_list = [60, 120, 240, 400, 800, 1600]
            args.N_Classes = [20, 20, 20, 20, 100, 100]
            args.sf_min_testerror = 0.6

        train_dataset = X_CIFAR(train_data, number_samples=50000)
        test_dataset = X_CIFAR(test_data, number_samples=10000)

    else:
        if args.dataset == 'MNIST':
            train_data = MNIST(args.data_path, train=True, download=True)
            test_data = MNIST(args.data_path, train=False, download=True)
        elif args.dataset == 'FMNIST':
            train_data = FashionMNIST(args.data_path, train=True, download=True)
            test_data = FashionMNIST(args.data_path, train=False, download=True)
        train_dataset = X_MNIST(train_data, number_samples=60000, dataset=args.dataset)
        test_dataset = X_MNIST(test_data, number_samples=10000, dataset=args.dataset)

    return args, architecture, preds, train_dataset, test_dataset

def configure_test(args):

    # # Evaluation - Manual # #


    if args.CFSE:
        architecture = 'CFSE'
    else:
        architecture = 'FFCNN'
    if args.sf_pred:
        preds = 'AvgSf'
    else:
        preds = 'Avg'

    if args.dataset[:5] == 'CIFAR':
        if args.dataset == 'CIFAR':
            print('CIFAR-10')
            train_data = CIFAR10(args.data_path, train=True, download=True)
            test_data = CIFAR10(args.data_path, train=False, download=True)
        else:
            print('CIFAR-100')
            train_data = CIFAR100(args.data_path, train=True, download=True)
            test_data = CIFAR100(args.data_path, train=False, download=True)
            args.ClassGroup = True
            args.channels_list = [60, 120, 240, 400, 800, 1600]
            args.N_Classes = [20, 20, 20, 20, 100, 100]
            args.sf_min_testerror = 0.6

        # train_dataset = X_CIFAR(train_data, number_samples=50000)
        test_dataset = X_CIFAR(test_data, number_samples=10000)

    else:
        if args.dataset == 'MNIST':
            train_data = MNIST(args.data_path, train=True, download=True)
            test_data = MNIST(args.data_path, train=False, download=True)
        elif args.dataset == 'FMNIST':
            train_data = FashionMNIST(args.data_path, train=True, download=True)
            test_data = FashionMNIST(args.data_path, train=False, download=True)
        # train_dataset = X_MNIST(train_data, number_samples=60000, dataset=args.dataset)
        test_dataset = X_MNIST(test_data, number_samples=10000, dataset=args.dataset)

    epoch = args.load_epoch
    loss_criterion = args.loss_criterion  # {'CwC', 'CwC_CE', 'PvN', 'CWG'}"
    dataset = args.dataset
    ILT = args.ILT  # {Acc, Fast}
    CFSE = args.CFSE  # {False = FFCNN}
    sf_pred = args.sf_pred
    ClassGroup = args.ClassGroup
    stage = args.stage
    batch_size = args.batch_size
    channels_list = args.channels_list
    N_Classes = args.N_Classes

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model_id = architecture + '_' + loss_criterion + '_' + preds + '_' + dataset + '_' + ILT + '_batch' + str(batch_size) + '_Ch' + 'n'.join(map(str, channels_list)) + '_stage' + str(stage)

    # LOAD MODEL
    if ClassGroup:
        model = CW_Comp_ClassGroup(channels_list, N_Classes, batch_size=batch_size, CFSE=CFSE, sf_pred=sf_pred, dataset=dataset, ILT=ILT, loss_=loss_criterion)
    else:
        model = CW_Comp(channels_list, batch_size=batch_size, CFSE=CFSE, sf_pred=sf_pred, dataset=dataset, ILT=ILT, loss_=loss_criterion, N_Classes=N_Classes)
    model = load_model(model, model_id, dataset, epoch)

    print(model)
    print(model_id)

    return model, test_loader
