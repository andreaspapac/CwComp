
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Models import *
from configure import *

CUDA_LAUNCH_BLOCKING = 1.
# torch.cuda.set_device(1)
print(torch.cuda.current_device())


if __name__ == "__main__":

    args = parse_args()
    model, test_loader = configure_test(args)
    sf_pred = args.sf_pred
    visualize_features = True
    #  Main Training Loop

    Avg_train_losses = []
    Avg_test_losses = []
    Sf_train_losses = []
    Sf_test_losses = []
    layerwise_loss = []

    epoch_errors = []
    sf_epoch_errors = []

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

    # print('Avg Pred - Train Error = {}'.format(np.asarray(epoch_losses).mean()))
    print('Avg Pred - Test Error = {}'.format(np.asarray(epoch_errors).mean()))
    if sf_pred:
        # print('Sf Pred -- Train Error = {}'.format(np.asarray(sf_epoch_losses).mean()))
        print('Sf Pred -- Test Error = {}'.format(np.asarray(sf_epoch_errors).mean()))