clc,clear;close all;
% solving off grid compressed sensing with ISTA
% Date: 2019.4.22  Editor: Yinchuan Li
%====================================================== Basic Parameters

i = sqrt(-1);
% N = 128;
% Ns = 64;
N=16
Ns = N/2;
J = 2;
L = N; % grid number

sigma_w = 0.0;
numiters = 1;
co=25;
avg_ista_off_norm = zeros(co,1);
tic
for it=1:numiters

% Signal Generate
% [y_t,phi,f,w,x_t] = GenOnGrid(N,Ns,J,L,sigma_w);
[y_t,phi,f,w,x_f] = GenOffGrid(N,Ns,J,L,sigma_w);
y_f = fft(y_t);
% disp(y_f)
% disp("done!!!!!!!!!!!!!!!!!!!")

% figure;plot(abs(y_t));
% figure;plot(abs(y_f));

% =========================================================
% on grid compressed sensing estimation
% tic
% [f_est,w_est,CS] = CS_grid(y_f,phi,sigma_w,L);
% toc
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
% =========================================================
% on grid ISTA estimation
% tic
% [f_est,w_est,ISTA] = ISTA_grid(y_f,phi,sigma_w,L);
% toc
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');



% tic
% [f_est,w_est,x_est,ISTA] = ISTA_flexgrid_Z(y_f,phi,sigma_w,L,w,f,x_f,J);
% toc


[f_est,w_est,x_est,norms,ISTA] = ISTA_flexgrid_Z_two(y_f,phi,sigma_w,L,w,f,x_f,J,co);
avg_ista_off_norm = norms+avg_ista_off_norm
disp(it)
disp('Done')
end
toc
avg_ista_off_norm = avg_ista_off_norm/numiters
figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
figure;plot(abs(x_f),'--or');hold on;plot(abs(x_est));
% disp(size(x_est))
annet_norms = [0.89193175,  1.2140325,   0.97708812,  0.70798879, 35.96070152,  0.5093414]
annet_len = length(annet_norms)
ista_len = length(avg_ista_off_norm)
ista_x = 1:1:ista_len
annet_x = 1:1:annet_len
figure;plot(ista_x,avg_ista_off_norm,'--bo');hold on;plot(annet_x,annet_norms,'--rs');
legend('ISTA off grid','AN-NET')
disp(avg_ista_off_norm)

i1 = [1.0438
    1.0184
    1.0096
    1.0231
    0.4402
    0.3634
    0.2176
    0.7319
    0.7387
    0.7387
    0.7388
    0.7388
    0.7387
    0.7388
    0.7388
    0.7387
    0.7387
    0.7378
    0.7378
    0.7378
    0.7378
    0.7378
    0.7378
    0.7378
    0.7378];
i2 = [1.0834
    0.5267
    0.2901
    0.2728
    0.0588
    0.1099
    0.0297
    0.0209
    0.0238
    0.0177
    0.0190
    0.0060
    0.0173
    0.0333
    0.0198
    0.0030
    0.0170
    0.0246
    0.0243
    0.8081
    0.8081
    0.8081
    0.8081
    0.8081
    0.8081];
i3 = [1.0603
    1.0335
    0.2966
    0.3996
    0.2618
    0.0752
    0.0379
    0.0154
    0.0422
    0.0162
    0.0323
    0.0050
    0.0128
    0.0156
    0.0067
    0.0314
    0.0388
    0.0386
    0.0184
    0.7567
    0.7567
    1.1381
    1.1381
    1.1381
    1.1381];
i4 = [0.7974
    0.8749
    0.8178
    0.8139
    0.1927
    0.0705
    0.0324
    0.0158
    0.0367
    0.0297
    0.0105
    0.0283
    0.0121
    0.0081
    0.0084
    0.0096
    0.0137
    0.0154
    0.0199
    0.7788
    0.7788
    0.7788
    0.7788
    0.7788
    0.7788];
i5 = [1.0036
    0.8494
    0.8170
    0.5136
    0.3192
    0.2388
    0.0723
    0.0239
    0.1256
    0.1395
    0.1381
    0.1418
    0.0276
    0.0629
    0.0548
    0.9069
    0.9069
    0.9069
    0.9069
    0.9069
    0.9069
    0.9073
    0.9073
    0.9073
    0.9073];

% NMSE = norm(x_est-x_f)/norm(x_f)



% Save data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
