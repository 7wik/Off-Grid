function [f_est,w_est,ISTA] = ISTA_flexgrid1(y_f,phi,sigma_w,L,w,f,x_f)

i = sqrt(-1);
N = length(y_f);
C = zeros(N,L);
co=10;%迭代次数
ex_dot=10000;%拓展点数
% f_set=zeros(co,10);
f_set=[];
%% first
inter=1;
dtribu=ones(1,L*ex_dot);
ff=nan;
ww=nan;

[dtribu,f_grid] = divideGrid(L,ff,ww,dtribu,inter);
% figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end
lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda,inter);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
%         ISTA.amp(m) = norm(ws(l));
        ISTA.amp(m) = ws(l);
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;plot(dtribu);hold on;stem(f*L,w*max(dtribu),'bo');hold on;stem(f_est*L,w_est*max(dtribu),'rs');
f_set{1,1}=f_est;

len=length(f_est);
for l=1:len-1
    temp=abs(f_est(l+1)-f_est(l));
    if temp>0.1
        break
    end
end

% w_nor(1:l)=abs(w_est(1:l))/sum(abs(w_est(1:l)));
% w_nor(l+1:len)=abs(w_est(l+1:len))/sum(abs(w_est(l+1:len)));
w_nor(1:l)=w_est(1:l)/sum(w_est(1:l));
w_nor(l+1:len)=w_est(l+1:len)/sum(w_est(l+1:len));
er1=0;
er2=0;
for ll=1:len
    if ll<=l
    er1=er1+f_est(ll)*w_nor(ll);
    else
    er2=er2+f_est(ll)*w_nor(ll);
    end
end

er{1,1}=er1-f(1);
er{1,2}=er2-f(2);


%% second--
disp(inter)
for ii=2:co
inter=inter+1;
% f_est=f;
% w_est=real(w);
[dtribu,f_grid] = divideGrid(L,f_est,w_est,dtribu,inter);
%figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end

lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda,inter);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
%         ISTA.amp(m) = norm(ws(l));
        ISTA.amp(m) = ws(l);
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;


f_set{ii,1}=f_est;
len=length(f_est);
for l=1:len-1
    temp=abs(f_est(l+1)-f_est(l));
    if temp>0.1
        break
    end
end

w_nor(1:l)=w_est(1:l)/sum(w_est(1:l));
w_nor(l+1:len)=w_est(l+1:len)/sum(w_est(l+1:len));
er1=0;
er2=0;
for ll=1:len
    if ll<=l
    er1=er1+f_est(ll)*w_nor(ll);
    else
    er2=er2+f_est(ll)*w_nor(ll);
    end
end

er{ii,1}=er1-f(1);
er{ii,2}=er2-f(2);

% figure;plot(dtribu);hold on;stem(f*L*ex_dot,w*max(dtribu),'bo');hold on;stem(f_est*L*ex_dot,w_est*max(dtribu),'rs');
% xlabel('f'),ylabel('w')
end
x_f1=x_f;
x_f2 = exp(-1i*2*pi*[0:1:N-1]'*f_est)*(w_est).';
% x_f2 = exp(-1i*2*pi*[0:1:N-1]'*f)*(w);
figure;plot(abs(x_f1),'--or');hold on;plot(abs(x_f2))
xx=1
