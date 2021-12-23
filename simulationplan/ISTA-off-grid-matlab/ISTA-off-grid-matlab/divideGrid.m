function [dtribu_new,f_grid] = divideGrid(L,f,w,dtribu,iter)
scar=3;
ex_dot=10000;
x=[0:1/ex_dot:L-1/ex_dot];
[Q,J]=size(f);
y=zeros(J,L*ex_dot);
sum_y=zeros(1,L*ex_dot);
f_grid=zeros(1,L);

% for l=1:J
%     w(l)=1;
% end

if iter==1
    sum_y=ones(1,L*ex_dot)*0.01;
else  
    for l=1:J
      miu=f(l)*L;
%       sig=w(l)*L/scar/2^iter;  %scar参数可调
%       y(l,:)= (sqrt(2*pi)*sig).^(-1) * exp(-(x-miu).^2/(2*sig*sig));
       sig=L/scar/(2^iter);  %scar参数可调
       y(l,:)=abs(w(l))*exp(-(x-miu).^2/(2*sig*sig));
      sum_y=sum_y+y(l,:);
    end
end

%figure;plot(dtribu);hold on;stem(f*L*ex_dot,w*max(dtribu),'bo');
dtribu_new=sum_y.*dtribu;
% dtribu_new=sum_y+dtribu;
%figure;plot(dtribu_new)
yi=dtribu_new;
% xi=(0:1/interpo/ex_dot:L-1/interpo/ex_dot);
% yi = interp1(x,dtribu_new,xi, 'spline'); 
S=sum(yi)/L;



temp=0;
count=1;
for i=1:L*ex_dot
  temp=temp+yi(i);
  if temp>=S
      temp=temp-S;
      
      f_grid(1,count)=i/L/ex_dot;
      count=count+1;
      if count>L
          break
      end
  end
end
% for i=count:L
%     f_gridnew(1,i)=f_gridnew(1,count-1)+(i-count+1)*(1-f_gridnew(1,count-1))/(L-count+1);
% end
if count==L
f_grid(1,L)=f_grid(1,L-1)+1/L/ex_dot;
end

% f_grid(1,L)=f_grid(1,L-1);

% figure;stem(f_grid,ones(1,L));
% xlabel('f'),ylabel('grid')

% figure;plot(f_gridnew)





end