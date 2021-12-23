function [f_est,w_est,ISTA] = ISTA_flexgrid(y_f,phi,sigma_w,L,w,f)

i = sqrt(-1);
N = length(y_f);
C = zeros(N,L);

%% first
inter=1;
dtribu=ones(1,L);
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
ws = lasso_ista(y_f,C,lambda);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;plot(dtribu);hold on;stem(f*64,w*max(dtribu),'bo');hold on;stem(f_est*64,w_est*max(dtribu),'rs');
ff1=f_est;

%% second
inter=inter+1;
[dtribu,f_grid] = divideGrid(L,f_est,w_est,dtribu,inter);
% figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end
lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
% figure;plot(dtribu);hold on;stem(f*64,w*max(dtribu),'bo');hold on;stem(f_est*64,w_est*max(dtribu),'rs');
ff2=f_est;

%% thrid
inter=inter+1;
[dtribu,f_grid] = divideGrid(L,f_est,w_est,dtribu,inter);
% figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end
lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
% figure;plot(dtribu);hold on;stem(f*64,w*max(dtribu),'bo');hold on;stem(f_est*64,w_est*max(dtribu),'rs');
ff3=f_est;

%% forth
inter=inter+1;
[dtribu,f_grid] = divideGrid(L,f_est,w_est,dtribu,inter);
% figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end
lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
% figure;plot(dtribu);hold on;stem(f*64,w*max(dtribu),'bo');hold on;stem(f_est*64,w_est*max(dtribu),'rs');
ff4=f_est;

%% fifth
inter=inter+1;
[dtribu,f_grid] = divideGrid(L,f_est,w_est,dtribu,inter);
% figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end
lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
% figure;plot(dtribu);hold on;stem(f*64,w*max(dtribu),'bo');hold on;stem(f_est*64,w_est*max(dtribu),'rs');
ff5=f_est;

%% sixth
inter=inter+1;
[dtribu,f_grid] = divideGrid(L,f_est,w_est,dtribu,inter);
% figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end
lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
% figure;plot(dtribu);hold on;stem(f*64,w*max(dtribu),'bo');hold on;stem(f_est*64,w_est*max(dtribu),'rs');
ff6=f_est;

%% seventh
inter=inter+1;
[dtribu,f_grid] = divideGrid(L,f_est,w_est,dtribu,inter);
% figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end
lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
% figure;plot(dtribu);hold on;stem(f*64,w*max(dtribu),'bo');hold on;stem(f_est*64,w_est*max(dtribu),'rs');
ff7=f_est;

%% eighth
inter=inter+1;
ss=dtribu;
[dtribu,f_grid] = divideGrid(L,f_est,w_est,dtribu,inter);
% figure;plot(ss*max(dtribu));hold on;plot(dtribu*max(ss),'rs');
% figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end
lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
% figure;plot(dtribu);hold on;stem(f*64,w*max(dtribu),'bo');hold on;stem(f_est*64,w_est*max(dtribu),'rs');
ff8=f_est;

%% ninth
inter=inter+1;
[dtribu,f_grid] = divideGrid(L,f_est,w_est,dtribu,inter);
% figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end
lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
% figure;plot(dtribu);hold on;stem(f*64,w*max(dtribu),'bo');hold on;stem(f_est*64,w_est*max(dtribu),'rs');
ff9=f_est;

%% tenth
inter=inter+1;
[dtribu,f_grid] = divideGrid(L,f_est,w_est,dtribu,inter);
% figure;stem(f_grid,ones(1,L));
%CS algorithm /Joint estimation
for l=1:L
    xs = exp(-i*2*pi*f_grid(l)*[0:1:N-1]');
    C(:,l) = phi.*xs;
end
lambda = 10/sqrt(2)*sigma_w*sqrt(2*log(L));
%ISTA
ws = lasso_ista(y_f,C,lambda);
%
ISTA.f = [];
ISTA.amp = [];
m=0;
for l=1:L
    if norm(ws(l))>=2e-2
        m = m+1;
        ISTA.f(m) = f_grid(l);
        ISTA.amp(m) = norm(ws(l));
    end
end
f_est = ISTA.f;
w_est = ISTA.amp;
% figure;stem(f,w,'bo');hold on;stem(f_est,w_est,'rs');
% figure;plot(dtribu);hold on;stem(f*64,w*max(dtribu),'bo');hold on;stem(f_est*64,w_est*max(dtribu),'rs');
ff10=f_est;

xx=1
