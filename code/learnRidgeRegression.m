function w = learnRidgeRegression(X,y,lambda)

% Implement ridge regression training here
% Inputs:
% X = N x D
% y = N x 1
% lambda = scalar
% Output:
% w = D x 1

Xtrans=X';%(DXN)
I=eye(size(X,2));%(DXD)
N=size(X,1);%(N)
A=N*lambda*I;%(DXD)
B=Xtrans*X;%(DXD)
C=Xtrans*y;%(NXD * NX1=DX1)
w=inv(A + B)*C;%(DX1)