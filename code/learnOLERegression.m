function w = learnOLERegression(X,Y)

% Implement OLE training here
% Inputs:
% X = N x D
% y = N x 1
% Output:
% w = D x 1

A=X' * X;%( DXN * NXD=DXD )
A=inv(A);%( DXD )
B=X' * Y;%( DXN * NX1=DX1)
w=A * B;%(DXD * DX1=DX1)