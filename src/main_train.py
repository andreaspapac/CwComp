
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Models import *
from configure import *

CUDA_LAUNCH_BLOCKING = 1.
# torch.cuda.set_device(1)
print(torch.cuda.current_device())


if __name__ == "__main__":

    args = parse_args()
    args, architecture, preds, train_dataset, test_dataset = configure(args)

    data_path = args.data_path
    seed = args.seed   # 52/13/22/2
    loss_criterion = args.loss_criterion  # {'CwC', 'CwC_CE', 'PvN', 'CWG'}"
    dataset = args.dataset
    ILT = args.ILT  # {Acc, Fast}
    save_ = args.save
    CFSE = args.CFSE  # {False = FFCNN}
    sf_pred = args.sf_pred
    ClassGroup = args.ClassGroup
    retrain = args.retrain
    stage = args.stage
    batch_size = args.batch_size
    channels_list = args.channels_list
    show_iters = args.show_iters
    n_epochs_d = args.n_epochs
    N_Classes = args.N_Classes
    min_testerror = args.min_testerror

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model_id = architecture + '_' + loss_criterion + '_' + preds + '_' + dataset + '_' + ILT + '_batch' + str(batch_size) + '_Ch' + 'n'.join(map(str, channels_list)) + '_stage' + str(stage)

    seed_everything(seed=seed)

    if ClassGroup:
        model = CW_Comp_ClassGroup(channels_list, N_Classes, batch_size=batch_size, CFSE=CFSE, sf_pred=sf_pred, dataset=dataset, ILT=ILT, loss_=loss_criterion)
    else:
        model = CW_Comp(channels_list, batch_size=batch_size, CFSE=CFSE, sf_pred=sf_pred, dataset=dataset, ILT=ILT, loss_=loss_criterion, N_Classes=N_Classes)

    print(model)
    print(model_id)
    if retrain:
        model = load_model(model, model_id, dataset, stage, param=True)

    #  Main Training Loop

    Avg_train_losses = []
    Avg_test_losses = []
    Sf_train_losses = []
    Sf_test_losses = []
    layerwise_loss = []

    for epoch in range(n_epochs_d):

        epoch_losses = []
        sf_epoch_losses = []
        ep_layer_l = []
        print('\n')
        print('-- Epoch: {} ------------------------------------'.format(epoch))

        model.train()
        for step, (x_p, y_p) in enumerate(train_loader):

            x_p = Variable(x_p).cuda()
            y_p = y_p.long()
            y_n = y_p[torch.randperm(x_p.size(0))]

            model.train_(x_p, y_p, y_n, epoch)

            if sf_pred:
                pred, sf = model.predict(x_p)
                sf_batch_loss = 1.0 - sf.eq(y_p.cuda()).float().mean().item()
                sf_epoch_losses.append(sf_batch_loss)
            else:
                pred = model.predict(x_p)

            batch_loss = 1.0 - pred.eq(y_p.cuda()).float().mean().item()
            epoch_losses.append(batch_loss)

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

                batch_loss = 1.0 - pred.eq(y_p.cuda()).float().mean().item()
                epoch_errors.append(batch_loss)

        # # PRINTS FOR EVERY EPOCH
        print('Epoch: {}'.format(epoch))
        print('Avg Pred - Train Error = {}'.format(np.asarray(epoch_losses).mean()))
        print('Avg Pred - Test Error = {}'.format(np.asarray(epoch_errors).mean()))
        if sf_pred:
            print('Sf Pred -- Train Error = {}'.format(np.asarray(sf_epoch_losses).mean()))
            print('Sf Pred -- Test Error = {}'.format(np.asarray(sf_epoch_errors).mean()))

        for i, layer in enumerate(model.conv_layers):
            layer_loss = layer.epoch_loss()
            ep_layer_l.append(layer_loss)
            print('Conv Layer_{} Loss : {}'.format(i, layer_loss))

        for i, layer in enumerate(model.nn_layers):
            layer_loss = layer.epoch_loss()
            ep_layer_l.append(layer_loss)
            print('          NN Layer_{} Loss : {}'.format(i, layer_loss))

        ep_layer_l.append(model.classifier_b1.epoch_loss())

        layerwise_loss.append(ep_layer_l)

        Avg_train_losses.append(np.asarray(epoch_losses).mean())
        Avg_test_losses.append(np.asarray(epoch_errors).mean())
        Sf_train_losses.append(np.asarray(sf_epoch_losses).mean())
        Sf_test_losses.append(np.asarray(sf_epoch_errors).mean())

        # SAVE MODEL
        if min_testerror > np.asarray(epoch_errors).mean() and save_:
            save_state(model, model_id, dataset, stage + 1)
            min_testerror = np.asarray(epoch_errors).mean()

    if sf_pred:
        pred_losses = [Avg_train_losses, Avg_test_losses, Sf_train_losses, Sf_test_losses]
    else:
        pred_losses = [Avg_train_losses, Avg_test_losses]    #, gd_train_losses, gd_test_losses]
    save_traininglog(layerwise_loss, model_id + 'layerwise_losses', layer_losses=True)
    save_traininglog(pred_losses, model_id + 'predictor_losses', layer_losses=False)

    torch.cuda.empty_cache()