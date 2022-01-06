#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: satwik
@version: 2020.03.12
Change to complex-valued network
Off-grid BBB ANNET net
Change the loss to y-loss
where in the network return x, and in the loss we use y-Cx
"""

import numpy as np
import matplotlib.pyplot as plt
from LISTA_E_BBB import my_lista
import random as rd
import torch
from scipy.special import comb, perm
batch_size = 2048
learning_rate = 1e-3
stage = 6
# dimensions of the sparse signal x
n = 8
# dimension of the compressed signal y
m = 4
# sparsity : non-zero values over n
sparsity = 2

# training and testing number:
train_number = 500000
valid_number = 500
# number of training iterations (epochs)
numIter = 1000

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


# generate off-grid training/testing data
def generate_off_grid_data(m, n, Phi, sparsity, bs, SNR=20):
    w_real = np.zeros([sparsity, bs])
    w_imag = np.zeros([sparsity, bs])
    Y_star = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
    Y = np.zeros((m, bs)) + 1j * np.zeros((m, bs))
    w_star = np.zeros([sparsity, bs]) + 1j * np.zeros([sparsity, bs])
    distance = 2/(n-1)
    f_interval = np.arange(0,sparsity*distance,distance)

    for i in range(bs):
        # f_star = (1-(sparsity-1)*distance) * np.sort(np.random.random((sparsity))) + f_interval
        f_star = np.sort(np.random.random((sparsity)))
        f_star = f_star.reshape(1,sparsity)
        Cf = np.exp(1j * 2 * np.pi * np.matmul(sample, f_star))

        w_real[:, i] = np.ones(sparsity)
        w_imag[:, i] = np.ones(sparsity)
        # w_real[:, i] = 2*np.random.random((sparsity))-1
        # w_imag[:, i] = 2*np.random.random((sparsity))-1
        w_star[:, i] = w_real[:, i] + 1j * w_imag[:, i]
        Y_star[:, i]  = np.matmul(Cf,w_star[:, i])
        Y[:, i] = np.matmul(Phi,Y_star[:, i])

    return Y, Y_star, f_star, w_star

train_input, train_output, f_star, w_star = generate_off_grid_data(m, n, Phi, sparsity, train_number)
valid_input, valid_output, f_star, w_star = generate_off_grid_data(m, n, Phi, sparsity, valid_number)

# print(y_train[:,1])
# print(x_train[:,1])

# compute the max eigen value of the w'*w
# alpha = np.linalg.norm(A,ord=2) ** 2
alpha = 5
# regularization paramter
_lambda = 0.5

SNR_list = []

Y_pred_real, Y_pred_imag, X_pred_real, X_pred_imag, err_list = my_lista(A, C, Phi, train_input, train_output, valid_input, valid_output,
                                               _lambda, alpha, bs=batch_size, epochs=numIter, stage = stage, learning_rate = learning_rate)
Y_pred_lista = Y_pred_real + 1j*Y_pred_imag


wt_gt_real = torch.from_numpy(w_star.real)
wt_gt_imag = torch.from_numpy(w_star.imag)

yt_gt_real = torch.from_numpy(valid_output.real)
yt_gt_imag = torch.from_numpy(valid_output.imag)

y_valid_input_real = torch.from_numpy(valid_input.real)
y_valid_input_imag = torch.from_numpy(valid_input.imag)


f_grid_plt = f_grid.reshape(n)
f_star_plt = f_star.reshape(sparsity)
for i in range(10):
    plt.figure()
    plt.subplot(211)
    plt.plot(f_grid_plt, X_pred_real[:, i], color='red', linestyle='--', marker='*', linewidth=1.5,
             label='LISTA')
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))

    plt.stem(f_star_plt, wt_gt_real.detach().numpy()[:, i], linefmt='b-', markerfmt='bo')

    plt.title("X Real Part")

    plt.subplot(212)
    plt.plot(f_grid_plt, X_pred_imag[:, i], color='red', linestyle='--', marker='*', linewidth=1.5,
             label='LISTA')
    plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
    plt.stem(f_star_plt, wt_gt_imag.detach().numpy()[:, i], linefmt='b-', markerfmt='bo')

    plt.title("X Imag Part")
    # plt.show()
    plt.savefig('X_plot.jpg')

    plt.figure()
    plt.subplot(211)
    plt.plot(Y_pred_real[:, i], color='red', linestyle='--', marker='*', linewidth=1.5,
             label='LISTA')
    plt.plot(yt_gt_real.detach().numpy()[:, i], color='green', linestyle='-', marker='+', linewidth=1.5,
             label='Ground_truth')
    plt.plot(y_valid_input_real.detach().numpy()[:, i], color='black', linestyle='-', marker='+', linewidth=1.5,
             label='input')
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("Y Real Part")

    plt.subplot(212)
    plt.plot(Y_pred_imag[:, i], color='red', linestyle='--', marker='*', linewidth=1.5,
             label='LISTA')
    plt.plot(yt_gt_imag.detach().numpy()[:, i], color='green', linestyle='-', marker='+', linewidth=1.5,
             label='Ground_truth')
    plt.plot(y_valid_input_imag.detach().numpy()[:, i], color='black', linestyle='-', marker='+', linewidth=1.5,
             label='input')
    plt.legend(loc='best', bbox_to_anchor=(0.2, 0.95))
    plt.title("Y Imag Part")
    # plt.show()
    plt.savefig('Y_plot.jpg')

