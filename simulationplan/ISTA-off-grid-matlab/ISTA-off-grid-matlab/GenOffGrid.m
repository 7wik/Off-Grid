function [yr,phi,f,w,x_f] = GenOffGrid(N,Ns,J,L,sigma_w)
% code generation
phi = zeros(N,1);
index = randperm(N,Ns);
phi(index) = 1;
% for l=1:N
%     if mod(l,3)==0
%       phi(l) = 1;
%     end
% end
    
w = ones(J,1).*(1+1i);

% Off-grid 
% Random frequency locations with a minimum separation of 2/(N-1)
distance = 2 ./ (N - 1);
f_interval = ( (distance/2) : distance : (J)*distance );
f = sort(rand(1,J))*0.9+ f_interval;%sort(A)若A是矩阵，默认对A的各列进行升序排列

% f=[0.1,0.8];
x_f = exp(-1i*2*pi*[0:1:N-1]'*f)*w;
x_t = ifft(x_f);

y_f = x_f.*phi;

% % Noise
% scr = 10^(scr_dB/10);
% Noise = (sigma_w/sqrt(2)).*(randn(N,1)+1i*randn(N,1));     %% noise
% scr = 10^(scr_dB/10);

%
yr = ifft(y_f);          % signal in time-domain

% disp(size(yr))
% disp(size(x_f))
% bre

end