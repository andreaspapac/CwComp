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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

sf_train_losses = []
sf_test_losses = []
gd_train_losses = []
gd_test_losses = []

layerwise_loss = []
batch_size = 128

n_repeats = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # int(round(60000/batch_size))+1
show_iters = 200
lr_decay = False

ILT = 'Acc'
dataset = 'CIFAR'  # MNIST/FMNIST/CIFAR
seed = 52          # 52/13/22/2

n_epochs_d = 50
model_id = 'ESANN_CWG_' + dataset + ILT

if dataset == 'CIFAR':
    train_data = CIFAR10('./data/', train=True, download=True)
    test_data = CIFAR10('./data/', train=False, download=True)
    train_dataset = Hybrid_CIFAR10(train_data, number_samples=50000)
    test_dataset = Hybrid_CIFAR10_test(test_data, number_samples=10000)
else:
    if dataset == 'MNIST':
        train_data = MNIST('./data/', train=True, download=True)
        test_data = MNIST('./data/', train=False, download=True)
    elif dataset == 'FMNIST':
        train_data = FashionMNIST('./data/', train=True, download=True)
        test_data = FashionMNIST('./data/', train=False, download=True)
    train_dataset = HybridImages_Train(train_data, number_samples=60000, dataset=dataset)
    test_dataset = HybridImages_Test(test_data, number_samples=10000, dataset=dataset)


if __name__ == "__main__":
    # torch.manual_seed(1234)
    seed_everything(seed=seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=5000, shuffle=False)

    # model = CFSE_CwC_2B(batch_size=batch_size, n_repeats=n_repeats, dataset=dataset, ILT=ILT)
    # model = FFCNN_PvN_4L(batch_size=batch_size, n_repeats=n_repeats, dataset=dataset, ILT=ILT)
    model = ESANN_CWG(batch_size=batch_size, n_repeats=n_repeats, dataset=dataset, ILT=ILT)
    print(model)

    for ep in range(n_epochs_d):
        epoch = ep * n_repeats[2]
        sf_epoch_losses = []
        gd_epoch_losses = []
        ep_layer_l = []
        print('\n')
        print('-- Epoch: {} ------------------------------------'.format(epoch))

        for step, ((x_p, y_p), (x_n, y_n)) in enumerate(train_loader):

            x_p = Variable(x_p).cuda()
            x_n = Variable(x_n).cuda()
            y_p = y_p.long()
            y_n = y_n.long()


            model.train(x_p, x_n, y_p, y_n, epoch)

            good, comb, sf_2 = model.predict(x_p)

            sf_batch_loss = 1.0 - sf_2.eq(y_p.cuda()).float().mean().item()
            sf_epoch_losses.append(sf_batch_loss)

            gd_batch_loss = 1.0 - good.eq(y_p.cuda()).float().mean().item()
            gd_epoch_losses.append(gd_batch_loss)

            # if step == len(train_loader) - 1:
            #     print('Iteration: {}/{}'.format(step, len(train_loader)))
            #     comb_acc = comb.eq(y_p.cuda()).float().mean().item()
            #     print('Combined Training accuracy: {}'.format(comb_acc))
            #     print('Goodness Predictor Training accuracy: {}'.format(1 - gd_batch_loss))
            #     print('---Ca_Sf Predictor Training accuracy:'.format(1 - sf_batch_loss))

        sf_epoch_errors = []
        gd_epoch_errors = []
        with torch.no_grad():
            for step, (x_p, y_p) in enumerate(test_loader):
                x_p = Variable(x_p).cuda()
                y_p = y_p.long()

                good, comb, sf_2 = model.predict(x_p)
                sf_batch_loss = 1.0 - sf_2.eq(y_p.cuda()).float().mean().item()
                sf_epoch_errors.append(sf_batch_loss)

                gd_batch_loss = 1.0 - good.eq(y_p.cuda()).float().mean().item()
                gd_epoch_errors.append(gd_batch_loss)

                comb_acc = comb.eq(y_p.cuda()).float().mean().item()
                print('Combined Testing accuracy: {}'.format(comb_acc))

        # # PRINTS FOR EVERY EPOCH
        print('Epoch: {}'.format(epoch))
        print('Gd Pred Train Error = {}'.format(np.asarray(gd_epoch_losses).mean()))
        print('-Gd Pred Test Error = {}'.format(np.asarray(gd_epoch_errors).mean()))
        print('Sf Pred Train Error = {}'.format(np.asarray(sf_epoch_losses).mean()))
        print('-Sf Pred Test Error = {}'.format(np.asarray(sf_epoch_errors).mean()))

        for i, layer in enumerate(model.convb1_layers):
            layer_loss = layer.epoch_loss()
            ep_layer_l.append(layer_loss)
            print('Block 1 Conv Layer_{} Loss : {}'.format(i, layer_loss))

        for i, layer in enumerate(model.convb2_layers):
            layer_loss = layer.epoch_loss()
            ep_layer_l.append(layer_loss)
            print('Block 2 Conv Layer_{} Loss : {}'.format(i, layer_loss))

        for i, layer in enumerate(model.nn_layers):
            layer_loss = layer.epoch_loss()
            ep_layer_l.append(layer_loss)
            print('          NN Layer_{} Loss : {}'.format(i, layer_loss))

        ep_layer_l.append(model.classifier_b1.epoch_loss())

        layerwise_loss.append(ep_layer_l)

        sf_train_losses.append(np.asarray(sf_epoch_losses).mean())
        sf_test_losses.append(np.asarray(sf_epoch_errors).mean())

        gd_train_losses.append(np.asarray(gd_epoch_losses).mean())
        gd_test_losses.append(np.asarray(gd_epoch_errors).mean())

        # SAVE MODEL
    #     if epoch > 10:
    #         save_model(model, model_id, dataset, epoch)
    #
    SFnGD_losses = [sf_train_losses, sf_test_losses, gd_train_losses, gd_test_losses]
    save_traininglog(layerwise_loss, model_id + 'layerwise_losses', layer_losses=True)
    save_traininglog(SFnGD_losses, model_id + 'predictor_losses', layer_losses=False)

    torch.cuda.empty_cache()

    # # SAVE PLOTS
    # plt.figure(1)
    # plt.plot(sf_train_losses, "k-^")
    # plt.plot(sf_test_losses, "k-*")
    # plt.plot(gd_train_losses, "b-^")
    # plt.plot(gd_test_losses, "b-*")
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.title("Train loss:--, Val loss:^ per epoch")
    # plt.savefig('figs/' + model_id + "epoch_losses_val.png")
    # plt.close(1)
    #
    # plt.figure(1)
    # styles = ['g-o', "g-*",'c-o', "c-*", 'r-o', 'r-*', 'y-o', 'y-*']
    # for i in range(np.shape(layerwise_loss)[1]):
    #     layer_losses = np.asarray(layerwise_loss)[:,i]
    #
    #     plt.plot(layer_losses, styles[i])
    #
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.title("Layers Losses per Epoch")
    # plt.savefig('figs/' + model_id + "Layers_epoch_losses.png")
    # plt.close(1)