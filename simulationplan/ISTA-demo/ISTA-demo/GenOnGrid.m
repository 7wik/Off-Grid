function [y_t,phi,f,w,x_t] = GenOnGrid(N,Ns,J,L,sigma_w)
% code generation
phi = zeros(N,1);
index = randperm(N,Ns);
phi(index) = 1;

w = ones(J,1).*(1+1i);

% on-grid frequency generation
f_grid = [0:1/L:1-1/L];
indexf = randperm(L,J);   %p = randperm(n,k) ����һ�д�1��n�������е�k����������k����Ҳ�ǲ���ͬ��
f = f_grid(indexf);
x_f = exp(-1i*2*pi*[0:1:N-1]'*f)*w;
x_t = ifft(x_f);

y_f = x_f.*phi;

% % Noise
% scr = 10^(scr_dB/10);
% Noise = (sigma_w/sqrt(2)).*(randn(N,1)+1i*randn(N,1));     %% noise
% scr = 10^(scr_dB/10);

%
y_t = ifft(y_f);          % signal in time-domain

end