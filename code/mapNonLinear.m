function x_n = mapNonLinear(x,d)
% Inputs:
% x - a single column vector (N x 1)
% d - integer (>= 0)
% Outputs:
% x_n - (N x (d+1))


for i=0:d
    x_n(i+1,:)=x.^i;
end
x_n=x_n';
