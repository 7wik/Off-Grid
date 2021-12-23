"""
@author: yinchuan li
@version: learn lambda and A version
@Data: 2020.03.18
Change to complex-valued network
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
import os
import time


# def soft_comp(x, threshold):
#     # hard thresholding
#     mask1 = x > threshold
#     mask2 = x < -threshold
#     out = torch.zeros_like(x)
#     out +=  mask1.float() * x
#     out +=  mask2.float() * x
#     return out


def soft_comp(x, threshold):
    mask1 = x > threshold
    mask2 = x < -threshold
    out = torch.zeros_like(x)
    out += mask1.float() * -threshold + mask1.float() * x
    out += mask2.float() * threshold + mask2.float() * x
    return out

# def soft_comp(x, threshold):
#     z = torch.zeros_like(x)
#     out = torch.sign(x) * torch.max(torch.abs(x) - threshold,z)
#     return out




class LISTA(nn.Module):
    def __init__(self, A, stage, alpha, thr):
        super(LISTA, self).__init__()
        self.alpha = alpha
        self.stage = stage
        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(stage):

            w_real = nn.Parameter(torch.from_numpy(A.real))
            w_imag = nn.Parameter(torch.from_numpy(A.imag))

            th = nn.Parameter(torch.tensor(thr))

            layer_params = (w_real, w_imag, th)

            # suffix = '_reverse' if direction == 1 else ''
            suffix = ''
            param_names = ['weights_real_{}{}', 'weights_imag_{}{}', 'threshold{}{}']

            param_names = [x.format(layer, suffix) for x in param_names]

            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._flat_weights_names.extend(param_names)
            self._all_weights.append(param_names)


    def forward(self, y_real, y_imag):

        param_names_init = self._all_weights[0][:]
        A_real = getattr(self, param_names_init[0])
        A_imag = getattr(self, param_names_init[1])
        thr = getattr(self, param_names_init[2])

        x_real = soft_comp(1 / self.alpha * (A_real.T.mm(y_real) + A_imag.T.mm(y_imag)), thr)
        x_imag = soft_comp(1 / self.alpha * (A_real.T.mm(y_imag) - A_imag.T.mm(y_real)), thr)

        for layer in range(self.stage):
            # x = F.softshrink(self.w1(y) + self.w2(x), self.thr)
            # x = self.soft(self.w1(y) + self.w2(x))
            param_names = self._all_weights[layer][:]
            A_real = getattr(self, param_names[0])
            A_imag = getattr(self, param_names[1])
            thr = getattr(self, param_names[2])

            Y_AX_real = y_real - (getattr(self, param_names[0]).mm(x_real) - getattr(self, param_names[1]).mm(x_imag))
            Y_AX_imag = y_imag - (getattr(self, param_names[0]).mm(x_imag) + getattr(self, param_names[1]).mm(x_real))

            z_org_real = x_real + 1 / self.alpha * (A_real.T.mm(Y_AX_real) + A_imag.T.mm(Y_AX_imag))
            z_org_imag = x_imag + 1 / self.alpha * (A_real.T.mm(Y_AX_imag) - A_imag.T.mm(Y_AX_real))
            x_real = soft_comp(z_org_real, thr)
            x_imag = soft_comp(z_org_imag, thr)
        return x_real, x_imag




def my_loss_y(A_real, A_imag, x_real, x_imag, y_real, y_imag, _lambda):
    criterion1 = nn.MSELoss()
    criterion2 = L1Loss_comp
    y_h_real = torch.mm(A_real, x_real) - torch.mm(A_imag, x_imag)
    y_h_imag = torch.mm(A_imag, x_real) + torch.mm(A_real, x_imag)
    # compute the losss
    loss1 = criterion1(y_h_real, y_real) + criterion1(y_h_imag, y_imag)
    loss2 = 0.1 * _lambda * criterion2(y_h_real, y_h_imag, y_real, y_imag)
    return loss1 + loss2


def my_loss_nmse(x_real, x_imag, x_gt_real, x_gt_imag):
    x_real[x_real < 1e-8] = 2e-8
    x_imag[x_real < 1e-8] = 2e-8
    x_gt_real[x_gt_real < 1e-8] = 1e-8
    x_gt_imag[x_gt_imag < 1e-8] = 1e-8
    x_nmse = 0
    for i in range(x_real.shape[1]):
        x_gt_norm = torch.sum(x_gt_real[:, i] ** 2 + x_gt_imag[:, i] ** 2)
        sub_norm = torch.sum((x_real[:, i] - x_gt_real[:, i]) ** 2 + (x_imag[:, i] - x_gt_imag[:, i]) ** 2)
        x_nmse += sub_norm / x_gt_norm
    return x_nmse / x_real.shape[1]


def my_loss_x(x_real, x_imag, x_gt_real, x_gt_imag, _lambda):
    criterion1 = nn.MSELoss()
    plot_flag = 0
    # x_real[x_real < 1e-8] = 2e-8
    # x_imag[x_real < 1e-8] = 2e-8
    # compute the losss
    loss1 = (criterion1(x_real, x_gt_real)  # *(x_gt_real.shape[0]*x_gt_real.shape[1])/ non_zero_real
             + criterion1(x_imag, x_gt_imag))  # *(x_gt_imag.shape[0]*x_gt_imag.shape[1])/ non_zero_imag


    if plot_flag:
        xi = x_imag.detach().numpy()[:, 0]
        xr = x_real.detach().numpy()[:, 0]
        xi_gt = x_gt_imag.detach().numpy()[:, 0]
        xr_gt = x_gt_real.detach().numpy()[:, 0]
        plt.figure()
        plt.subplot(311)
        plt.stem(range(xi.shape[0]), xi, use_line_collection=True, markerfmt='D', label='LISTA')
        plt.stem(range(xi_gt.shape[0]), xi_gt, use_line_collection=True, markerfmt='o', label='Ground_truth')
        plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
        plt.title("LISTA-Net Real Part")

        plt.subplot(312)
        plt.stem(range(xr.shape[0]), xr, markerfmt='D', label='LISTA')
        plt.stem(range(xr_gt.shape[0]), xr_gt, use_line_collection=True, markerfmt='o', label='Ground_truth')
        plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
        plt.title("LISTA-Net Imag Part")

        plt.subplot(313)
        plt.stem(range(xi.shape[0]), np.sqrt(xi ** 2 + xr ** 2), use_line_collection=True, markerfmt='D', label='LISTA')
        plt.stem(range(xi_gt.shape[0]), np.sqrt(xi_gt ** 2 + xr_gt ** 2), use_line_collection=True, markerfmt='o',
                 label='Ground_truth')
        plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
        plt.title("LISTA-Net Imag Part")
        plt.show()

    return loss1


def L1Loss_comp(input_real, input_imag, target_real, target_imag):
    return torch.mean(torch.sqrt((input_real - target_real) ** 2 + (input_imag - target_imag) ** 2))



def now_to_date(format_string="%Y-%m-%d-%H-%M-%S"):
    time_stamp = int(time.time())
    time_array = time.localtime(time_stamp)
    str_date = time.strftime(format_string, time_array)
    return str_date


def save_my_data(y_train, x_train, y_valid, x_valid, path_data='data/LISTA_Train_Valid_Data/'):
    print('===> Saving data...')
    str_date = now_to_date()
    data = {
        'train_data': y_train,
        'train_label': x_train,
        'valid_data': y_valid,
        'valid_label': x_valid}
    if not os.path.isdir(path_data):
        os.makedirs(path_data)
    torch.save(data, path_data + 'data-' + str_date + '.pt')
    print('===> Saving data finished...')


def save_my_model(net, optimizer, epochs, loss_list, loss_val_list, learning_rate, stage,
                  path_checkpoint='model/LISTA-Update-Weights/checkpoint/'):
    print('===> Saving models...')
    str_date = now_to_date()
    state = {
        'epoch': epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_list,
        'loss_val': loss_val_list,
        'learning_rate': learning_rate,
        'stage': stage
    }
    # path_checkpoint = 'model/LISTA-Updata-Weights/checkpoint/'
    if not os.path.isdir(path_checkpoint):
        os.makedirs(path_checkpoint)
    torch.save(state, path_checkpoint + 'LISTA_net-' + str_date + '.pt')
    print('===> Saving model finished...')


def my_lista(Phi, C, A, y_train, x_train, y_valid, x_valid, _lambda, alpha, bs, epochs, stage, learning_rate):
    nt = y_train.shape[-1]
    nv = y_valid.shape[-1]
    PRINT_FLAG = 0
    # batch_size = y_train.shape[-1]

    # _w2_ = (np.eye(A.shape[-1]) - (1 / alpha) * np.matmul(A.conj().T, A))  # note that here is the conjugate transpose
    # _w1_ = ((1 / alpha) * A.conj().T)
    _Phi_ = Phi
    _C_ = C
    # map
    # convert the data into tensors
    y_real = torch.from_numpy(y_train.real)
    y_imag = torch.from_numpy(y_train.imag)
    x_gt_real = torch.from_numpy(x_train.real)
    x_gt_imag = torch.from_numpy(x_train.imag)
    yt_real = torch.from_numpy(y_valid.real)
    yt_imag = torch.from_numpy(y_valid.imag)
    xt_gt_real = torch.from_numpy(x_valid.real)
    xt_gt_imag = torch.from_numpy(x_valid.imag)

    # list of losses at every epoch
    loss_list = []
    loss_val_list = []
    # ------- Training phase --------------
    # number of layers:

    net = LISTA(A, stage, alpha, thr=_lambda / alpha)

    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # momentum=0.9
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, momentum=0.9) # momentum=0.9
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9) #

    for epoch in range(epochs):

        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

        for i in range((nt - 1) // bs + 1):
            # get the outputs
            start_i = i * bs
            end_i = start_i + bs
            x_real, x_imag = net(y_real[:, start_i:end_i], y_imag[:, start_i:end_i])

            loss = my_loss_x(x_real, x_imag, x_gt_real[:, start_i:end_i], x_gt_imag[:, start_i:end_i], _lambda)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print('Epoch: %d mini-batch: %d  train-loss: %.5f ' % (epoch, i + 1, loss.detach().data))
            loss_list.append(loss.detach().data)
        with torch.no_grad():
            for i in range((nv - 1) // (2 * bs) + 1):
                start_i = i * (2 * bs)
                end_i = start_i + (2 * bs)
                x_pred_real, x_pred_imag = net(yt_real[:, start_i:end_i], yt_imag[:, start_i:end_i])

                loss_val = my_loss_nmse(x_pred_real, x_pred_imag,
                                        xt_gt_real[:, start_i:end_i], xt_gt_imag[:, start_i:end_i])
                # if i % 10 == 0:
                #     print('Epoch: %d mini-batch: %d  valid-loss %.2f'%(epoch, i, loss_val.detach().data))
                loss_val_list.append(loss_val.detach().data)
            print('Epoch: %d  mean-valid-loss %.5f' % (epoch, np.mean(loss_val_list)))

    save_my_model(net, optimizer, epochs, loss_list, loss_val_list, learning_rate, stage,
                  path_checkpoint='model/LISTA-Update-Weights/checkpoint/')
    # save_my_data(y_train, x_train, y_valid, x_valid, path_data='data/LISTA_Train_Valid_Data/')

    if PRINT_FLAG:
        plt.figure()
        plt.subplot(211)
        plt.plot(x_pred_real.detach().numpy()[:, 0], color='red', linestyle='--', marker='*', linewidth=1.5,
                 label='LISTA')
        plt.plot(xt_gt_real.detach().numpy()[:, 0], color='green', linestyle='-', marker='+', linewidth=1.5,
                 label='Ground_truth')
        plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
        plt.title("LISTA-Net Real Part")

        plt.subplot(212)
        plt.plot(x_pred_imag.detach().numpy()[:, 0], color='red', linestyle='--', marker='*', linewidth=1.5,
                 label='LISTA')
        plt.plot(xt_gt_imag.detach().numpy()[:, 0], color='green', linestyle='-', marker='+', linewidth=1.5,
                 label='Ground_truth')
        plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
        plt.title("LISTA-Net Imag Part")
        plt.show()
    return x_pred_real.numpy(), x_pred_imag.numpy(), loss_list


