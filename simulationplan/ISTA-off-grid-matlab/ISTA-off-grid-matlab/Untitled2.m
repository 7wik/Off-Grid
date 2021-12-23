clc,clear;close all;
% L=1024;
% J=2;
% w = ones(J,1).*(1+1i);
% f_grid = [0:1/L:1-1/L];
% indexf = randperm(L,J);   %p = randperm(n,k) 返回一行从1到n的整数中的k个，而且这k个数也是不相同的
% f = f_grid(indexf);
% f=[0.1,0.9];
% f_gridnew=zeros(1,L);
% 
% x=[0:1:L-1];
% y=zeros(J,L);
% sum_y=zeros(1,L);
% inter=zeros(1,L);
% 
% for l=1:L
% miu=f(l)*L;
% sig=abs(w(l))*L/10;
% y(l,:)= (sqrt(2*pi)*sig).^(-1) * exp(-(x-miu).^2/(2*sig*sig));
% sum_y=sum_y+y(l,:);
% end
% for l=1:L
% inter(1,l)=1.0/sum_y(1,l);
% end
% k=1.0/sum(inter);
% 
% for l=2:L
% f_gridnew(:,l)=f_gridnew(:,l-1)+inter(:,l)*k;
% end
% 
% % figure;plot(x,y(1,:));
% % figure;plot(x,y(2,:));
% figure;plot(x/1024,sum_y);
% figure;stem(f_gridnew,ones(1,L));
