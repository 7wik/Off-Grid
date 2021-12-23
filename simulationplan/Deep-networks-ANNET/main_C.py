#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yinchuan li
@version: 2020.03.12
Change to complex-valued network
Change the loss to y-loss
where in the network return x, and in the loss we use y-Cx
"""

import numpy as np
import matplotlib.pyplot as plt
from LISTA_C import my_lista
import random as rd
import torch
from scipy.special import comb, perm
batch_size = 50
learning_rate = 1e-5
stage = 10
# dimensions of the sparse signal x
n = 10
# dimension of the compressed signal y
m = 5
# sparsity : non-zero values over n
sparsity = 2

# training and testing number:
train_number = 10000
valid_number = 100
# number of training iterations (epochs)
numIter = 500

# w state of randomness
# rng = np.random.RandomState(23)


# create the 1:N sample
sample = np.array(list(range(m))).reshape(m, 1)
# create the frequency grid
f_grid = 1/n*np.array(list(range(n))).reshape(1, n)

# generate the dictionary, without the sampling matrix Phi
# C = np.exp(1j*2*np.pi*np.matmul(sample, f_grid))
# A = C.copy()

# generate the dictionary with the sampling matrix Phi
C = np.exp(1j*2*np.pi*np.matmul(sample, f_grid))
phi = np.zeros([m])
p = 0.8 # sparsity level
Ns = int(np.floor((p)*m)) # the sampling number, note that should be int
index = rd.sample(range(m),Ns)
phi[index] = 1
Phi = np.diag(phi)
A = np.matmul(Phi,C)

# generate training/testing data
def generate_data(A, C, Phi, sparsity, bs, SNR=20):
    m, n = A.shape
    x_real = np.zeros([n, bs])
    x_imag = np.zeros([n, bs])
    x = np.zeros([n, bs]) + 1j * np.zeros([n, bs])
    for i in range(bs):
        idx = rd.sample(list(range(n)), sparsity)

        x_real[idx, i] = np.ones(sparsity)
        x_imag[idx, i] = np.ones(sparsity)
        temp = x_real[:, i] + 1j * x_imag[:, i]
        x[:, i] = temp                                     # without the normalization of x
        # x[:, i] = temp / np.linalg.norm(temp)            # with the normalization of x
    X = x.copy()

    # ----- Generate the compressed noiseless signals --------
    Y = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
    Y_star = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
    for i in range(bs):
        tmp = np.matmul(A, X[:, i])
        # tmp /= np.linalg.norm(tmp)
        Y[:, i] = tmp
        Y_star[:,i] = np.matmul(C, X[:, i])

    # ----- Generate the compressed signals with noise -------
    # noise_std = np.power(10, -(SNR / 20))
    # Y = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
    # for i in range(bs):
    #     tmp = np.matmul(A, X[:, i])
    #     # tmp /= np.linalg.norm(tmp)
    #     Y[:, i] = tmp + noise_std * (rng.randn(m) + 1j * rng.randn(m))
    return Y, Y_star, X

train_input, train_output, x_train = generate_data(A, C, Phi, sparsity, train_number)
valid_input, valid_output, x_valid = generate_data(A, C, Phi, sparsity, valid_number)

# print(y_train[:,1])
# print(x_train[:,1])

# compute the max eigen value of the w'*w
# alpha = np.linalg.norm(A,ord=2) ** 2
alpha = 5
# regularization paramter
_lambda = 0.5

SNR_list = []

X_pred_real, X_pred_imag, err_list = my_lista(A, C, Phi, train_input, train_output, valid_input, valid_output,
                                               _lambda, alpha, bs=batch_size, epochs=numIter, stage = stage, learning_rate = learning_rate)
X_pred_lista = X_pred_real + 1j*X_pred_imag


xt_gt_real = torch.from_numpy(x_valid.real)
xt_gt_imag = torch.from_numpy(x_valid.imag)

yt_gt_real = torch.from_numpy(valid_output.real)
yt_gt_imag = torch.from_numpy(valid_output.imag)

y_valid_input_real = torch.from_numpy(valid_input.real)
y_valid_input_imag = torch.from_numpy(valid_input.imag)

C_real = torch.from_numpy(C.real)
C_imag = torch.from_numpy(C.imag)
y_h_real = np.matmul(C_real, X_pred_real) - np.matmul(C_imag, X_pred_imag)
y_h_imag = np.matmul(C_imag, X_pred_real) + np.matmul(C_real, X_pred_imag)

for i in range(valid_number):
    plt.figure()
    plt.subplot(411)
    plt.plot(X_pred_real[:, i], color='red', linestyle='--', marker='*', linewidth=1.5,
             label='LISTA')
    plt.plot(xt_gt_real.detach().numpy()[:, i], color='green', linestyle='-', marker='+', linewidth=1.5,
             label='Ground_truth')
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("LISTA-Net Real Part")

    plt.subplot(412)
    plt.plot(X_pred_imag[:, i], color='red', linestyle='--', marker='*', linewidth=1.5,
             label='LISTA')
    plt.plot(xt_gt_imag.detach().numpy()[:, i], color='green', linestyle='-', marker='+', linewidth=1.5,
             label='Ground_truth')
    plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
    plt.title("LISTA-Net Imag Part")


    plt.subplot(413)
    plt.plot(y_h_real[:, i], color='red', linestyle='--', marker='*', linewidth=1.5,
             label='LISTA')
    plt.plot(yt_gt_real.detach().numpy()[:, i], color='green', linestyle='-', marker='+', linewidth=1.5,
             label='Ground_truth')
    plt.plot(y_valid_input_real.detach().numpy()[:, i], color='black', linestyle='-', marker='+', linewidth=1.5,
             label='input')
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("LISTA-Net Real Part")

    plt.subplot(414)
    plt.plot(y_h_imag[:, i], color='red', linestyle='--', marker='*', linewidth=1.5,
             label='LISTA')
    plt.plot(yt_gt_imag.detach().numpy()[:, i], color='green', linestyle='-', marker='+', linewidth=1.5,
             label='Ground_truth')
    plt.plot(y_valid_input_imag.detach().numpy()[:, i], color='black', linestyle='-', marker='+', linewidth=1.5,
             label='input')
    plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
    plt.title("LISTA-Net Imag Part")
    plt.show()

