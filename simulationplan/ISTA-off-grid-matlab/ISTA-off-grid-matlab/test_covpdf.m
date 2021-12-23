clc;clear all;close all;
L = 16;
x = zeros(16,1);
x(randperm(16,5)) = rand(5,1);


ex_dot=10000;
s=[0:1/ex_dot:L-1/ex_dot];
sig = 0.5;
miu=1;
pdf = exp(-(s-miu).^2/(2*sig*sig));

figure;plot(pdf)