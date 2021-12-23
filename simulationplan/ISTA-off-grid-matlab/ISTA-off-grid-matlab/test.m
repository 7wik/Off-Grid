
clc,clear;close all;

figure;stem(tau_targ,f_targ,amp_targ,':bo','LineWidth',2);hold on;
stem(freq(2,:),freq(1,:),abs(amp),'-rs','LineWidth',2);
grid on;
axis([0,1,0,1]);
xlabel('\tau'),ylabel('f')
legend('Ground truth','AN','best');
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0.1 0.1 5 4]);
set(gca,'FontSize',12,'Layer','top','LineWidth',1);
set(gcf,'papersize',[5 4]);
saveas(gcf,'2D_AN.pdf');