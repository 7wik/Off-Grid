clc,clear;close all;
% solving on grid compressed sensing with ISTA
% Date: 2019.4.22  Editor: Yinchuan Li
%====================================================== Basic Parameters

i = sqrt(-1);
N = 64
Ns = N/2;
J = 2;
L = N; % grid number

sigma_w = 0.01;

% Signal Generate
[y_t,phi,f,w,x_t] = GenOnGrid(N,Ns,J,L,sigma_w);

y_f = fft(y_t);

% figure;plot(abs(y_t));
% figure;plot(abs(y_f));

% =========================================================
% on grid compressed sensing estimation
tic
[f_est,w_est,CS] = CS_grid(y_f,phi,sigma_w,L);
toc
figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');hold on;
% =========================================================
% on grid ISTA estimation
tic
[f_est2,w_est2,ISTA] = ISTA_grid(y_f,phi,sigma_w,L);
toc
% figure;stem(f,w,'bo');hold on;
stem(f_est2,w_est2,'m^');hold on;



