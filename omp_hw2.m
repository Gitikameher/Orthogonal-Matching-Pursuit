Prob_success=zeros(20);
for M=1:20
    for k=1:20
        count=0;count1=0;
        for iter=1:200
A=gen_random_A(M,20);
[s x]=generate_sparse_x(20,k);
noise=transpose(generate_noise(M,0.001));
y=A*x;
y=y+noise;
[n t r]=OMP(y,A,k);
o=calculate_x(t,A, n,y);
err=calculate_error(x,o);
if err<0.001
   count=count+1; 
end
Prob_success(k,M)=count/200;
        end
    end
end

function y=gen_random_A(m,n)
y=normc(randn(m,n));
return 
end

function [s y]=generate_sparse_x(n,p)
s=randperm(n, p);
y=zeros(1,n);
for i=1:length(s)
    t=randi([0 1],1);
    if t==1
        y(s(i))=unifrnd(-10, -1, 1, 1);
    else
        y(s(i))=unifrnd(1, 10, 1, 1);
    end
end
y=transpose(y);
return 
end

function y=generate_noise(n,s_dev)
y=randn(1,n)*s_dev;
return 
end

function e=calculate_error(x,x_hat)
y=x_hat-x;
e=sqrt(transpose(y)*y)/sqrt(transpose(x)*x);
return 
end
function x=calculate_x(ind, A, s, y)
[r c]=size(A);
x =zeros(c, 1);
z=(pinv((transpose(s))*s))*(transpose(s))*y;
x(ind)=z;
return
end

function [l x val]=OMP(y, A, k)
r_0=y;z_vector=zeros(size(y));
Set_S=[];ind_x=[];v=[];
for i=1:k
[Set_S ind_x v]=calculate_basis(r_0, Set_S, A, ind_x, v);
r_0=update_residual(Set_S, y);
u=calculate_x(ind_x, A, Set_S, y);
m=transpose(y-(A*u))*(y-A*u);
if r_0==z_vector
    break
end
end
l=Set_S;x=ind_x;val=v;
return
end

function [s x_i val_i]=calculate_basis(residual, set_S, A, set_x, var_i)
[w i]=max(abs(transpose(A)*residual));
w=transpose(A(:,i))*residual;
s=[set_S A(:,i)];
x_i=[set_x i];
val_i=[var_i w];
return
end

function r=update_residual(S, y)
P=S*((pinv(transpose(S)*S))*transpose(S));
[rows columns]=size(P);
r=(eye(rows)-P)*y;
return 
end