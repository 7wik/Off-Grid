"""
Author: Kartheek
Date: 29th Jan, 2020 ; 5:22 PM

This file builds and trains the Learned ISTA algorithm
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
import os
import time

def soft_comp(x, threshold):
    mask1 = x >  threshold
    mask2 = x < -threshold
    out = torch.zeros_like(x)
    out += mask1.float() * -threshold + mask1.float() * x
    out += mask2.float() * threshold + mask2.float() * x
    return out


class LISTA(nn.Module):
    def __init__(self, _w1_, _w2_, stage, alpha, thr):
        super(LISTA, self).__init__()
        # self.w_real = nn.Parameter(torch.from_numpy(_w2_.real))
        # self.w_imag = nn.Parameter(torch.from_numpy(_w2_.imag))
        # self.w1_real = nn.Parameter(torch.from_numpy(_w1_.real))
        # self.w1_imag = nn.Parameter(torch.from_numpy(_w1_.imag))
        # self.thershold = nn.Parameter(torch.tensor(thr))
        self.alpha = alpha
        # self.load_state_dict
        # self.lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=1)
        self.stage = stage
        self._flat_weights_names = []
        self._all_weights = []
        self.weights_real =  nn.ParameterList([])
        self.weights_imag =  nn.ParameterList([])
        self.bias_real =  nn.ParameterList([])
        self.bias_imag =  nn.ParameterList([])
        self.threshold =  nn.ParameterList([])
        # self.conv2d_1 = nn.ParameterList([])
        # self.conv2d_2 = nn.ParameterList([])
        for layer in range(stage):
            # w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
            # w_hh = Parameter(torch.Tensor(gate_size, hidden_size))

            # w_real = nn.Parameter(torch.from_numpy(_w2_.real))
            # w_imag = nn.Parameter(torch.from_numpy(_w2_.imag))
            # b_real = nn.Parameter(torch.from_numpy(_w1_.real))
            # b_imag = nn.Parameter(torch.from_numpy(_w1_.imag))
            # th = nn.Parameter(torch.tensor(thr))
            self.weights_real.append(nn.Parameter(torch.from_numpy(_w2_.real)))
            self.weights_imag.append(nn.Parameter(torch.from_numpy(_w2_.imag)))
            self.bias_real.append(nn.Parameter(torch.from_numpy(_w1_.real)))
            self.bias_imag.append(nn.Parameter(torch.from_numpy(_w1_.imag)))
            self.threshold.append(nn.Parameter(torch.tensor(thr)))
            # self.conv2d_1.append(nn.Conv2d(1, 3, 3))
            # self.conv2d_2.append(nn.Conv2d(3, 1, 3))
        # print(1)
            conv2d_1 = nn.Conv1d(1, 3, 3, padding=1) # 输入数据6*6 - meaning inputdata
            # # 加上一层relu - meaning = Add a layer
            conv2d_2 = nn.Conv1d(3, 1, 3, padding=1) # output channel is one


            layer_params = (conv2d_1, conv2d_2)
            suffix = ''
            # param_names = ['weights_real_{}{}', 'weights_imag_{}{}',
            #                'bias_real_{}{}', 'bias_imag_{}{}',
            #                'conv2d_1_{}{}','conv2d_2_{}{}',
            #                'threshold{}{}']
            param_names = ['conv2d_1_{}{}', 'conv2d_2_{}{}']
            param_names = [x.format(layer, suffix) for x in param_names]
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._flat_weights_names.extend(param_names)
            self._all_weights.append(param_names)

        # self._flat_weights = [getattr(self, weight) for weight in self._flat_weights_names]
        # self.flatten_parameters()

    # def flatten_parameters(self):
    #     any_param = next(self.parameters()).data
    #     if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
    #         return





    def forward(self, y_real, y_imag):

        # x_real = soft_comp(self.w1_real.mm(y_real) - self.w1_imag.mm(y_imag), self.threshold)
        # x_imag = soft_comp(self.w1_real.mm(y_imag) + self.w1_imag.mm(y_real), self.threshold)
        '''
                # param_names = ['weights_real_{}{}', 'weights_imag_{}{}',
                #                'bias_real_{}{}', 'bias_imag_{}{}', 'threshold{}{}']
                # w2_real, w2_imag, w1_real, w1_imag, threshold
        '''
        # param_names_init = self._all_weights[0][:]
        #
        # x_real = soft_comp(getattr(self, param_names_init[2]).mm(y_real) - \
        #                    getattr(self, param_names_init[3]).mm(y_imag), getattr(self, param_names_init[4]))
        # x_imag = soft_comp(getattr(self, param_names_init[2]).mm(y_imag) + \
        #                     getattr(self, param_names_init[3]).mm(y_real), getattr(self, param_names_init[4]))
        x_real = soft_comp(self.bias_real[0].mm(y_real) - \
                           self.bias_imag[0].mm(y_imag), self.threshold[0])
        x_imag = soft_comp(self.bias_real[0].mm(y_imag) + \
                            self.bias_imag[0].mm(y_real), self.threshold[0])

        for layer in range(self.stage):

            param_names = self._all_weights[layer][:]
            # w_real = getattr(self, param_names[0])
            # w_imag = getattr(self, param_names[0])
            # b_real = getattr(self, param_names[0])
            conv2d_1 = getattr(self, param_names[0])
            conv2d_2 = getattr(self, param_names[1])
	        # w_real = nn.Parameter(torch.from_numpy(_w2_.real))
	        # w_imag = nn.Parameter(torch.from_numpy(_w2_.imag))
	        # b_real = nn.Parameter(torch.from_numpy(_w1_.real))
	        # b_imag = nn.Parameter(torch.from_numpy(_w1_.imag))
	        # conv2d_1 = nn.Conv2d(1, 3, 3)  # 输入数据6*6
	        # # 加上一层relu
	        # conv2d_2 = nn.Conv2d(3, 1, 3)  # output channel is one
	        # th = nn.Parameter(torch.tensor(thr))
            # x_cov_real = self.conv2d_1[layer](x_real.view(()))



            z_org_real = self.weights_real[layer].mm(x_real) - \
                         self.weights_imag[layer].mm(x_imag) + \
                         self.bias_real[layer].mm(y_real) - \
                         self.bias_imag[layer].mm(y_imag)

            z_org_imag = self.weights_real[layer].mm(x_imag) + \
                         self.weights_imag[layer].mm(x_real) + \
                         self.bias_real[layer].mm(y_imag) + \
                         self.bias_imag[layer].mm(y_real)

            print(z_org_real.size(),"before conv layer +++++++++++++++++++++++++++")

            z_conv_real = (conv2d_1(z_org_real.T.view(64, 1, 8)))

            z_conv_imag = (conv2d_1(z_org_imag.T.view(64, 1, 8)))
            print(z_conv_real.size(),"after conv layer 1 ----------------------------")

            x_real = soft_comp(z_conv_real, self.threshold[layer])
            x_imag = soft_comp(z_conv_imag, self.threshold[layer])
            x_real = (conv2d_2(x_real)).view(64, 8).T
            x_imag = (conv2d_2(x_imag)).view(64, 8).T
            print(z_conv_real.size(),"after conv layer 2 ********************************")

            # x_real = soft_comp(x_real, self.threshold[layer])
            # x_imag = soft_comp(x_imag, self.threshold[layer])


            # z_org_real = self.w1_real.mm(y_real) - self.w1_imag.mm(y_imag) + \
            #              self.w2_real.mm(x_real) - self.w2_imag.mm(x_imag)
            # z_org_imag = self.w1_real.mm(y_imag) + self.w1_imag.mm(y_real) + \
            #              self.w2_real.mm(x_imag) + self.w2_imag.mm(x_real)

            # x_real = soft_comp(z_org_real, self.thershold)
            # x_imag = soft_comp(z_org_imag, self.thershold)
            # assert x_real.requires_grad == True
            # assert x_imag.requires_grad == True

        return x_real, x_imag

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, X, Y):
#         self.Y = Y
#         self.X = X
#
#     def __len__(self):
#         return self.Y.shape[1]
#
#     def __getitem__(self, idx):
#         return self.X[:, idx], self.Y[:, idx]


def my_loss_y(A_real,A_imag, x_real,x_imag, y_real, y_imag, _lambda):
    criterion1 = nn.MSELoss()
    criterion2 = L1Loss_comp
    y_h_real = torch.mm(A_real, x_real) - torch.mm(A_imag, x_imag)
    y_h_imag = torch.mm(A_imag, x_real) + torch.mm(A_real, x_imag)
    # compute the losss
    loss1 = criterion1(y_h_real, y_real) + criterion1(y_h_imag, y_imag)
    loss2 = 0.1 * _lambda * criterion2(y_h_real, y_h_imag, y_real, y_imag)
    return loss1 + loss2
def my_loss_nmse( x_real, x_imag, x_gt_real, x_gt_imag):
    x_real[x_real < 1e-8] = 2e-8
    x_imag[x_real < 1e-8] = 2e-8
    x_gt_real[x_gt_real < 1e-8] = 1e-8
    x_gt_imag[x_gt_imag < 1e-8] = 1e-8
    x_nmse = 0
    for i in range(x_real.shape[1]):
        x_gt_norm = torch.sqrt(torch.sum(x_gt_real[:, i]**2 + x_gt_imag[:, i]**2))
        sub_norm = torch.sqrt(torch.sum((x_real[:, i] - x_gt_real[:, i])**2 + (x_imag[:, i] - x_gt_imag[:, i])**2 ))
        x_nmse += sub_norm/x_gt_norm
    return   x_nmse / x_real.shape[1]

def my_loss_x( x_real, x_imag, x_gt_real, x_gt_imag, A, _lambda):
    criterion1 = nn.MSELoss()
    plot_flag = 0
    criterion2 = L1Loss_comp
    A = torch.ones(x_real.shape[0],x_real.shape[0],dtype=torch.float64)
    # y_h_real = torch.mm(A, x_real)
    # y_h_imag = torch.mm(A, x_imag)
    # y_gt_real = torch.mm(A, x_gt_real)
    # y_gt_imag = torch.mm(A, x_gt_imag)
    # x_real[x_real < 1e-8] = 1e-8
    # x_imag[x_real < 1e-8] = 1e-8
    # # non_zero_real = torch.sum(x_gt_real > 1e-8)
    # # non_zero_imag = torch.sum(x_gt_imag > 1e-8)
    # x_gt_real[x_gt_real < 1e-8] = 1e-8
    # x_gt_imag[x_gt_imag < 1e-8] = 1e-8

    # compute the losss
    loss1 = (criterion1(x_real, x_gt_real) #*(x_gt_real.shape[0]*x_gt_real.shape[1])/ non_zero_real
             + criterion1(x_imag, x_gt_imag))#*(x_gt_imag.shape[0]*x_gt_imag.shape[1])/ non_zero_imag

    # loss2 = 0.1 * _lambda * criterion2(x_real, x_imag, x_gt_real, x_gt_imag)
    # import matplotlib.pyplot as plt
    # plt.figure(11)
    # plt.subplot(211)
    # plt.plot(x_real.detach().numpy()[:,0], color='red', linestyle='--', marker='*', linewidth=1.5, label='LISTA')
    # plt.plot(x_gt_real.detach().numpy()[:,0], color='green', linestyle='-', marker='+', linewidth=1.5, label='Ground_truth')
    # plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    # plt.title("LISTA-Net Real Part")
    #
    # plt.subplot(212)
    # plt.plot(x_imag.detach().numpy()[:,0], color='red', linestyle='--', marker='*', linewidth=1.5, label='LISTA')
    # plt.plot(x_gt_imag.detach().numpy()[:,0], color='green', linestyle='-', marker='+', linewidth=1.5, label='Ground_truth')
    # plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
    # plt.title("LISTA-Net Imag Part")
    # plt.show()
    if plot_flag:
        xi = x_imag.detach().numpy()[:, 0]
        xr = x_real.detach().numpy()[:, 0]
        xi_gt = x_gt_imag.detach().numpy()[:, 0]
        xr_gt = x_gt_real.detach().numpy()[:, 0]
        plt.figure()
        plt.subplot(311)
        plt.stem( xi, use_line_collection=True, markerfmt='D', label='LISTA')
        plt.stem(xi_gt, use_line_collection=True, markerfmt='o', label='Ground_truth')
        plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
        plt.title("LISTA-Net Real Part")

        plt.subplot(312)
        plt.stem( xr, markerfmt='D', label='LISTA')
        plt.stem( xr_gt, use_line_collection=True, markerfmt='o', label='Ground_truth')
        plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
        plt.title("LISTA-Net Imag Part")

        plt.subplot(313)
        plt.stem( np.sqrt(xi**2+xr**2), use_line_collection=True, markerfmt='D', label='LISTA')
        plt.stem( np.sqrt(xi_gt**2+xr_gt**2), use_line_collection=True, markerfmt='o', label='Ground_truth')
        plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
        plt.title("LISTA-Net Imag Part")
        plt.show()

    return loss1 #+ loss2


def L1Loss_comp(input_real, input_imag, target_real, target_imag):
    return torch.mean(torch.sqrt((input_real - target_real)**2 + (input_imag - target_imag)**2 ))
# def L2Loss_comp(input_real, input_imag, target_real, target_imag):
#     return ((input_real - target_real)**2 + (input_imag - target_imag)**2 ))

def now_to_date(format_string="%Y-%m-%d-%H-%M-%S"):
    time_stamp = int(time.time())
    time_array = time.localtime(time_stamp)
    str_date = time.strftime(format_string, time_array)
    return str_date


def save_my_data(y_train, x_train, y_valid, x_valid, path_data='data/LISTA_Train_Valid_Data/'):
    print('===> Saving models...')
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


def save_my_model(net, optimizer, epochs, loss_list,loss_val_list, learning_rate, stage,
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
    print('===> Saving data finished...')


def my_lista(A, y_train, x_train, y_valid, x_valid, _lambda, alpha, bs, epochs,stage, learning_rate):
    # load the data to ensure every training has the same data

    # data_load = torch.load('data\LISTA_Train_Valid_Data\data-2020-03-07-00-19-22.pt')
    # y_train = data_load['train_data']
    # x_train = data_load['train_label']
    # y_valid = data_load['valid_data']
    # x_valid = data_load['valid_label']
    _lambda = 1.0
    nt = y_train.shape[-1]
    nv = y_valid.shape[-1]
    PRINT_FLAG = 1
    # batch_size = y_train.shape[-1]

    _w2_ = (np.eye(A.shape[-1]) - (1/alpha)*np.matmul(A.conj().T, A))  # 注意这里是共轭转置
    _w1_ = ((1/alpha)*A.conj().T)
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
    # the dictionary A
    rng = np.random.RandomState(23)
    A_seed = torch.from_numpy(rng.randn(x_gt_real.shape[0], x_gt_real.shape[0]))
    # A_real = torch.from_numpy(A.real)
    # A_imag = torch.from_numpy(A.imag)

    net = LISTA(_w1_, _w2_, stage, alpha=alpha, thr=_lambda / alpha)
    net = net.double()
    params = list(net.parameters())
    # for param in params:
    #     print(param.grad)
    # Update the weights

    # build the optimizer and criterion

    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # momentum=0.9
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, momentum=0.9) # momentum=0.9
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9) #

    # list of losses at every epoch
    loss_list = []
    loss_val_list = []
    # ------- Training phase --------------
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # model_load = torch.load('model\LISTA-Update-Weights\checkpoint\LISTA_net-2020-03-07-00-19-38.pt')
    # net.load_state_dict(model_load['model_state_dict'])
    # optimizer.load_state_dict(model_load['optimizer_state_dict'])
    # loss_list = model_load['loss']
    # loss_val_list = model_load['loss_val']
    for epoch in range(epochs):
        # if epoch < 100:
        #     learning_rate = 1e-3
        # elif epoch < 200:
        #     learning_rate = 0.9e-3
        # else:
        #     learning_rate = 0.5e-3

        # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  #
        for i in range((nt - 1) // bs +1):
            # get the outputs
            start_i = i * bs
            end_i = start_i + bs
            x_real, x_imag = net(y_real[:, start_i:end_i], y_imag[:, start_i:end_i])
            # loss =  my_loss(A_real, A_imag, x_real, x_imag, y_real, y_imag, _lambda)
            loss = my_loss_x(x_real, x_imag, x_gt_real[:, start_i:end_i], x_gt_imag[:, start_i:end_i], A_seed,_lambda)
            # loss = my_loss_nmse(x_real, x_imag, x_gt_real[:, start_i:end_i], x_gt_imag[:, start_i:end_i])
            # compute the gradiettns
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                print('Epoch: %d mini-batch: %d  train-loss: %.5f ' % (epoch, i+1, loss.detach().data))
            loss_list.append(loss.detach().data)
        with torch.no_grad():
            for i in range((nv - 1) // (2*bs) +1):
                start_i = i * (2*bs)
                end_i = start_i + (2*bs)
                x_pred_real, x_pred_imag = net(yt_real[:, start_i:end_i], yt_imag[:, start_i:end_i])
                loss_val = my_loss_x(x_pred_real, x_pred_imag,
                                     xt_gt_real[:, start_i:end_i], xt_gt_imag[:, start_i:end_i],
                                     A_seed,_lambda)
                # loss_val = my_loss_nmse(x_pred_real, x_pred_imag,
                #                      xt_gt_real[:, start_i:end_i], xt_gt_imag[:, start_i:end_i])
                # if i % 10 == 0:
                #     print('Epoch: %d mini-batch: %d  valid-loss %.2f'%(epoch, i, loss_val.detach().data))
                loss_val_list.append(loss_val.detach().data)
            print('Epoch: %d  mean-valid-loss %.5f'%(epoch, np.mean(loss_val_list)))

    save_my_model(net, optimizer, epochs, loss_list, loss_val_list, learning_rate, stage,
                  path_checkpoint='model/LISTA-Update-Weights/checkpoint/')

    if PRINT_FLAG:
        plt.figure()
        plt.subplot(211)
        plt.plot(x_pred_real.detach().numpy()[:, 0], color='red', linestyle='--', marker='*', linewidth=1.5, label='LISTA')
        plt.plot(xt_gt_real.detach().numpy()[:, 0], color='green', linestyle='-', marker='+', linewidth=1.5, label='Ground_truth')
        plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
        plt.title("LISTA-Net Real Part")

        plt.subplot(212)
        plt.plot(x_pred_imag.detach().numpy()[:, 0], color='red', linestyle='--', marker='*', linewidth=1.5, label='LISTA')
        plt.plot(xt_gt_imag.detach().numpy()[:, 0], color='green', linestyle='-', marker='+', linewidth=1.5, label='Ground_truth')
        plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
        plt.title("LISTA-Net Imag Part")
        plt.show()
    return x_pred_real.numpy(), x_pred_imag.numpy(), loss_list






# 保存模型示例代码- meaning = Save the model sample code
