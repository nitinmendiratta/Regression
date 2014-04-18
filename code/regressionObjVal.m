function [error, error_grad] = regressionObjVal(w, X, y, lambda)

% compute squared error (scalar) and gradient of squared error with respect
% to w (vector) for the given data X and y and the regularization parameter
% lambda
N=size(X,1);
A= X*w;%( 242x65 * 65X1=242x1)
B=(y - A).^2;%(242x1 - 242x1=242x1)
C=lambda*(w'*w);%1X1
error=(1/N)*(sum(B))+C;

%gradient of error
N=size(X,1);

deviation=X*w - y;%242x1 - 242x1=242x1
A = (2/N) * deviation' * X;%1x242 * 242x65=1x65
B = lambda * ( w');%(1x65)
error_grad = A + B;%(1x65)
error_grad = error_grad';%(65x1)
