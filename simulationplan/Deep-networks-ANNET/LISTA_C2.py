"""
@author: yinchuan li
@version: 2020.03.12
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


def soft_comp(x, threshold):
    mask1 = x > threshold
    mask2 = x < -threshold
    out = torch.zeros_like(x)
    out += mask1.float() * -threshold + mask1.float() * x
    out += mask2.float() * threshold + mask2.float() * x
    return out


class LISTA(nn.Module):
    def __init__(self, _w1_, _w2_, C, stage, alpha, thr):
        super(LISTA, self).__init__()
        self.alpha = alpha
        self.stage = stage
        self._flat_weights_names = []
        self._all_weights = []
        self.n = _w2_.shape[0]
        self.C_real = nn.Parameter(torch.from_numpy(C.real))
        self.C_imag = nn.Parameter(torch.from_numpy(C.imag))
        for layer in range(stage):

            w_real = nn.Parameter(torch.from_numpy(_w2_.real))
            w_imag = nn.Parameter(torch.from_numpy(_w2_.imag))
            b_real = nn.Parameter(torch.from_numpy(_w1_.real))
            b_imag = nn.Parameter(torch.from_numpy(_w1_.imag))
            th = nn.Parameter(torch.tensor(thr))
            layer_params = (w_real, w_imag, b_real, b_imag, th)
            suffix = ''
            param_names = ['weights_real_{}{}', 'weights_imag_{}{}',
                           'bias_real_{}{}', 'bias_imag_{}{}', 'threshold{}{}']
            param_names = [x.format(layer, suffix) for x in param_names]
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._flat_weights_names.extend(param_names)
            self._all_weights.append(param_names)


    def forward(self, y_real, y_imag):
        self.bs = y_real.shape[-1]
        '''
                # param_names = ['weights_real_{}{}', 'weights_imag_{}{}',
                #                'bias_real_{}{}', 'bias_imag_{}{}', 'threshold{}{}']
                # w2_real, w2_imag, w1_real, w1_imag, threshold
        '''

        x_real = torch.zeros([self.n, self.bs],dtype=torch.float64)
        x_imag = torch.zeros([self.n, self.bs],dtype=torch.float64)

        # from the first layer to the last layer, where in the first layer x=0
        for layer in range(self.stage):
            param_names = self._all_weights[layer][:]
            z_org_real = getattr(self, param_names[0]).mm(x_real) - \
                         getattr(self, param_names[1]).mm(x_imag) + \
                         getattr(self, param_names[2]).mm(y_real) - \
                         getattr(self, param_names[3]).mm(y_imag)

            z_org_imag = getattr(self, param_names[0]).mm(x_imag) + \
                         getattr(self, param_names[1]).mm(x_real) + \
                         getattr(self, param_names[2]).mm(y_imag) + \
                         getattr(self, param_names[3]).mm(y_real)

            x_real = soft_comp(z_org_real, getattr(self, param_names[4]))
            x_imag = soft_comp(z_org_imag, getattr(self, param_names[4]))

        Cx_real = self.C_real.mm(x_real) - self.C_imag.mm(x_imag)
        Cx_imag = self.C_imag.mm(x_real) + self.C_real.mm(x_imag)
        return Cx_real, Cx_imag

    # the following forward function has the same performance with the above one
    # def forward(self, y_real, y_imag):
    #     self.bs = y_real.shape[-1]
    #     '''
    #             # param_names = ['weights_real_{}{}', 'weights_imag_{}{}',
    #             #                'bias_real_{}{}', 'bias_imag_{}{}', 'threshold{}{}']
    #             # w2_real, w2_imag, w1_real, w1_imag, threshold
    #     '''
    #     # the first layer, where x = 0 and hence no x
    #     param_names_init = self._all_weights[0][:]
    #     x_real = soft_comp(getattr(self, param_names_init[2]).mm(y_real) - \
    #                        getattr(self, param_names_init[3]).mm(y_imag), getattr(self, param_names_init[4]))
    #     x_imag = soft_comp(getattr(self, param_names_init[2]).mm(y_imag) + \
    #                        getattr(self, param_names_init[3]).mm(y_real), getattr(self, param_names_init[4]))
    #
    #     # from the second layer to the last layer
    #     for layer in range(self.stage-1):
    #         param_names = self._all_weights[layer+1][:]
    #         z_org_real = getattr(self, param_names[0]).mm(x_real) - \
    #                      getattr(self, param_names[1]).mm(x_imag) + \
    #                      getattr(self, param_names[2]).mm(y_real) - \
    #                      getattr(self, param_names[3]).mm(y_imag)
    #
    #         z_org_imag = getattr(self, param_names[0]).mm(x_imag) + \
    #                      getattr(self, param_names[1]).mm(x_real) + \
    #                      getattr(self, param_names[2]).mm(y_imag) + \
    #                      getattr(self, param_names[3]).mm(y_real)
    #
    #         x_real = soft_comp(z_org_real, getattr(self, param_names[4]))
    #         x_imag = soft_comp(z_org_imag, getattr(self, param_names[4]))
    #
    #     return x_real, x_imag


def my_loss_y(A_real, A_imag, x_real, x_imag, y_real, y_imag):
    criterion1 = nn.MSELoss()
    # criterion2 = L1Loss_comp
    y_h_real = torch.mm(A_real, x_real) - torch.mm(A_imag, x_imag)
    y_h_imag = torch.mm(A_imag, x_real) + torch.mm(A_real, x_imag)
    # compute the losss
    loss1 = criterion1(y_h_real, y_real) + criterion1(y_h_imag, y_imag)
    # loss2 = 0.1 * _lambda * criterion2(y_h_real, y_h_imag, y_real, y_imag)
    return loss1


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


def my_loss_x(x_real, x_imag, x_gt_real, x_gt_imag):
    criterion1 = nn.MSELoss()
    plot_flag = 0

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


def my_lista(A, C, Phi, y_train, x_train, y_valid, x_valid, _lambda, alpha, bs, epochs,stage, learning_rate):
    nt = y_train.shape[-1]
    nv = y_valid.shape[-1]
    PRINT_FLAG = 0
    # batch_size = y_train.shape[-1]

    _w2_ = (np.eye(A.shape[-1]) - (1 / alpha) * np.matmul(A.conj().T, A))  # note that here is the conjugate transpose
    _w1_ = ((1 / alpha) * A.conj().T)
    # map
    # convert the data into tensors
    C_real = torch.from_numpy(C.real)
    C_imag = torch.from_numpy(C.imag)
    y_real = torch.from_numpy(y_train.real)
    y_imag = torch.from_numpy(y_train.imag)
    y_gt_real = torch.from_numpy(x_train.real)
    y_gt_imag = torch.from_numpy(x_train.imag)
    yt_real = torch.from_numpy(y_valid.real)
    yt_imag = torch.from_numpy(y_valid.imag)
    yt_gt_real = torch.from_numpy(x_valid.real)
    yt_gt_imag = torch.from_numpy(x_valid.imag)

    # list of losses at every epoch
    loss_list = []
    loss_val_list = []
    # ------- Training phase --------------
    # number of layers:
    # stage = 10
    net = LISTA(_w1_, _w2_, C, stage, alpha=alpha, thr=_lambda / alpha)

    # learning_rate = 0.01
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # momentum=0.9
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, momentum=0.9) # momentum=0.9
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9) #

    for epoch in range(epochs):

        if epoch < 100:
            learning_rate = 1e-4
        elif epoch < 200:
            learning_rate = 1e-5
        else:
            learning_rate = 1e-6

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # momentum=0.9

        for i in range((nt - 1) // bs + 1):
            # get the outputs
            start_i = i * bs
            end_i = start_i + bs
            x_real, x_imag = net(y_real[:, start_i:end_i], y_imag[:, start_i:end_i])

            loss = my_loss_x(x_real, x_imag, y_gt_real[:, start_i:end_i], y_gt_imag[:, start_i:end_i])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 50 == 0:
                print('Epoch: %d mini-batch: %d  train-loss: %.5f ' % (epoch, i + 1, loss.detach().data))
            loss_list.append(loss.detach().data)
        with torch.no_grad():
            for i in range((nv - 1) // (2 * bs) + 1):
                start_i = i * (2 * bs)
                end_i = start_i + (2 * bs)
                x_pred_real, x_pred_imag = net(yt_real[:, start_i:end_i], yt_imag[:, start_i:end_i])

                loss_val = my_loss_x(x_pred_real, x_pred_imag,
                                        yt_gt_real[:, start_i:end_i], yt_gt_imag[:, start_i:end_i])
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


