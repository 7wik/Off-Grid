function [poles,coeffs,Z] = callProny(s,prony_tol)

n=length(s);
m=ceil(n/2);

for i=1:m
    X(:,i)=s(i:m+i-1);
end
[U,E]=svd(X);
e = diag(E);
K=sum(e>max(e)*prony_tol);

% [p, q] = size(X);
% mdl = zeros(1, p-1);
% for idx=1:p-1
%     d0 = prod(e(idx+1:end)+0.1);
%     n0 = 1/(p-idx)*sum((e(idx+1:end)).^((p-idx)));
%     mdl(idx) = q*(log(n0/d0))+ 0.5*idx*(2*p-idx)*log(q);
% end
% [~, K]=min(mdl);

for i=1:m
    Y(:,i)=s(i:K+i);
end
R=Y*Y';
C=R(2:end,2:end);
a=inv(C)*(-R(2:end,1));

z = roots([1;a]);
Z=[];
for i=1:length(z)
    Z=[Z,z(i).^([0:1:n-1]')];
end
coeffs = inv(Z'*Z)*Z'*s;
poles = mod(atan2(imag(z),real(z)),2*pi);