#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yinchuan li
@version: 2020.03.12
Change to complex-valued network
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd
import torch
from scipy.special import comb, perm
# number of layers in the neural network
n_layers = 4
# learning_rate
lr = 1e-3
# training and testing number:
train_number = 50000
valid_number = 100
# batch size
batch_size = 128
# number of training iterations (epochs)
numIter = 1000

# dimensions of the sparse signal x
n = 8
# dimension of the compressed signal y
m = 4
# sparsity : non-zero values over n
sparsity = 2

# w state of randomness
# rng = np.random.RandomState(23)

# create the 1:N sample
sample = np.array(list(range(m))).reshape(m, 1)
# create the frequency grid
f_grid = 1/n*np.array(list(range(n))).reshape(1, n)

# ========= generate the dictionary, without the sampling matrix Phi
# C = np.exp(1j*2*np.pi*np.matmul(sample, f_grid))
# A = C.copy()

# ========= generate the dictionary with the sampling matrix Phi
# --------- method 1:
# C = np.exp(1j*2*np.pi*np.matmul(sample, f_grid))
# phi = np.ones([m])
# p = 0.8 # sparsity level
# Ns = int(np.floor((1-p)*m)) # the sampling number, note that should be int
# index = rd.sample(range(m),Ns)
# phi[index] = 0
# A = C.copy()
# for i in range(n):
#     A[index, i] = 0
#     # print(A[index, i])
# --------- method 2:
C = np.exp(1j*2*np.pi*np.matmul(sample, f_grid))
phi = np.zeros([m])
p = 0.8 # sparsity level
Ns = int(np.floor((p)*m)) # the sampling number, note that should be int
index = rd.sample(range(m),Ns)
phi[index] = 1
Phi = np.diag(phi)
A = np.matmul(Phi,C)


# generate training/testing data
def generate_data(A, sparsity, bs, SNR=20):
    m, n = A.shape
    x_real = np.zeros([n, bs])
    x_imag = np.zeros([n, bs])
    x = np.zeros([n, bs]) + 1j * np.zeros([n, bs])
    for i in range(bs):
        idx = rd.sample(list(range(n)), sparsity)
        # x_gt_real[idx, i] = rng.randn(sparsity)
        # x_gt_imag[idx, i] = rng.randn(sparsity)
        x_real[idx, i] = np.ones(sparsity)
        x_imag[idx, i] = np.ones(sparsity)
        temp = x_real[:, i] + 1j * x_imag[:, i]
        x[:, i] = temp                                     # without the normalization of x
        # x[:, i] = temp / np.linalg.norm(temp)            # with the normalization of x
    X = x.copy()

    # ----- Generate the compressed noiseless signals --------
    Y = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
    for i in range(bs):
        tmp = np.matmul(A, X[:, i])
        # tmp /= np.linalg.norm(tmp)
        Y[:, i] = tmp

    # ----- Generate the compressed signals with noise -------
    # noise_std = np.power(10, -(SNR / 20))
    # Y = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
    # for i in range(bs):
    #     tmp = np.matmul(A, X[:, i])
    #     # tmp /= np.linalg.norm(tmp)
    #     Y[:, i] = tmp + noise_std * (rng.randn(m) + 1j * rng.randn(m))
    return Y, X

y_train, x_train = generate_data(A, sparsity, train_number)
y_valid, x_valid = generate_data(A, sparsity, valid_number)



# compute the max eigen value of the w'*w
# alpha = np.linalg.norm(A,ord=2) ** 2 # approximate = 16
alpha = 5
# step_size = 1/alpha
# step_size = 0.1                       # approximate = 1/alpha
# regularization paramter
_lambda = 0.5
# theta = _lambda/alpha
# theta = 0.1

SNR_list = []

# ========= using learn W1 and W2 version
# from LISTA import my_lista
# X_pred_real, X_pred_imag, err_list = my_lista(A, y_train, x_train, y_valid, x_valid,
#                                                _lambda, alpha, bs=50, epochs=numIter)

# ========= using learn lambda and rho version
from LISTA_A import my_lista
X_pred_real, X_pred_imag, err_list = my_lista(Phi, C, A, y_train, x_train, y_valid, x_valid,
                                              _lambda = _lambda, alpha = alpha, bs=batch_size, epochs=numIter, stage = n_layers, learning_rate = lr)
X_pred_lista = X_pred_real + 1j*X_pred_imag


xt_gt_real = torch.from_numpy(x_valid.real)
xt_gt_imag = torch.from_numpy(x_valid.imag)


for i in range(valid_number):
    plt.figure()
    plt.subplot(211)
    plt.plot(X_pred_real[:, i], color='red', linestyle='--', marker='*', linewidth=1.5,
             label='LISTA')
    plt.plot(xt_gt_real.detach().numpy()[:, i], color='green', linestyle='-', marker='+', linewidth=1.5,
             label='Ground_truth')
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("LISTA-Net Real Part")

    plt.subplot(212)
    plt.plot(X_pred_imag[:, i], color='red', linestyle='--', marker='*', linewidth=1.5,
             label='LISTA')
    plt.plot(xt_gt_imag.detach().numpy()[:, i], color='green', linestyle='-', marker='+', linewidth=1.5,
             label='Ground_truth')
    plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
    plt.title("LISTA-Net Imag Part")
    plt.show()

