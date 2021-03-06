clc,clear;
close all;

i = sqrt(-1);

L =2;    
N = 24;
Ns = N/2;
sigma_w = 0 %sqrt(0.001);  % **** power of noise
rho = 5;   %
lambda = sigma_w*sqrt(N*log(N))/4;   %
gamma = lambda/sqrt(N);
mu = 0.1;  %
iterations = 100
avg = 0


% b = rand(N,1);
% b = ones(N,1);



tau = [0.21;0.29]; %[0.21;0.23;0.29]
% tau = [0.43];
amp = ones(L,1);
for i=1:iterations


b = zeros(N,1);
b(randperm(N,Ns)) = 1;

x = zeros(N,1);
for l=1:L
    x = x + amp(l)*exp(-i*2*pi*tau(l)*[0:1:N-1]');
end
yr = x.*b;

% yr = zeros(N,1);
% for l=1:L
%     yr = yr + amp(l)*exp(-i*2*pi*tau(l)*[0:1:N-1]').*b;
% end

% noise = (sigma_w/sqrt(2))*(randn(N,1)+i*randn(N,1));
noise = zeros(N,1);
y = yr + noise;
% figure;plot(abs(yr),'-b','LineWidth',2);hold on;
% plot(abs(noise),'-k','LineWidth',2);hold on;
% plot(abs(y),'-r','LineWidth',2);hold on;

% grid on;
% % axis([0,150,0,500]);
% xlabel('sample'),ylabel('magnitude');
% legend('echo','noise','echo+noise','location','best');
% set(gcf, 'PaperPositionMode', 'manual');
% set(gcf, 'PaperUnits', 'inches');
% set(gcf, 'PaperPosition', [0.1 0.1 5 4]);
% set(gca,'FontSize',12,'Layer','top','LineWidth',1);
% set(gcf,'papersize',[5 4]);
% % saveas(gcf,'signal2.pdf');
% saveas(gcf,'signal3.pdf');


% index_temp = call_Tgindex(N-1);
% 
% index_Tg1 = zeros(N-1,N-1);
% index_Tg2 = zeros(N-1,N-1);
% for i = 1:N-1
%     index_Tg1(i,1:N-i) = index_temp{1,i};
%     index_Tg2(i,1:N-i) = index_temp{2,i};
% end

fL = 0.2;
fH = 0.3;

[x1,tau_cvx1,amp_cvx1] = call1DAN(y,b,lambda);
avg = avg+norm(x-x1)/norm(x);
% disp(x1)
% disp()
end

disp(avg/iterations)
disp("+++++++++++++++++++++")



% [x] = call1D_FSAN(y,b,lambda,fL,fH,index_Tg1,index_Tg2);
% [x,tau_cvx2,amp_cvx2] = call1D_FSAN_v3(y,b,lambda,fL,fH,index_Tg1,index_Tg2);

figure;
stem(tau,abs(amp),'-bo','LineWidth',2);hold on;
stem(tau_cvx1,abs(amp_cvx1),'-rs','LineWidth',2);
% stem(tau_cvx2,abs(amp_cvx2),'-kx','LineWidth',2);

grid on;
axis([0,1,0,1.2]);
xlabel('delay'),ylabel('magnitude');
legend('ground truth','AN','location','best');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0.1 0.1 5 4]);
set(gca,'FontSize',12,'Layer','top','LineWidth',1);
set(gcf,'papersize',[5 4]);
saveas(gcf,'delay2.pdf');