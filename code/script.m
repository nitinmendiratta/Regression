% load the data
load diabetes;
x_train_i = [ones(size(x_train,1),1) x_train];
x_test_i = [ones(size(x_test,1),1) x_test];
%%% FILL CODE FOR PROBLEM 1 %%%

% linear regression without intercept
w=learnOLERegression(x_train,y_train);

%calculate the training error using RSE
deviation=(y_train - (x_train*w)).^2;%(242x1 - 242x1=242x1)
err_sum=sum(deviation);
error_train=sqrt(err_sum);
%calculate the testing error using RSE
deviation=(y_test - (x_test*w)).^2;%(200x1 - 200x1=200x1)
err_sum=sum(deviation);
error_test=sqrt(err_sum);

% linear regression with intercept
w=learnOLERegression(x_train_i,y_train);

%calculate the training error using RSE
deviation=(y_train - (x_train_i * w)).^2;%(242x1 - 242x1=242x1)
err_sum=sum(deviation);
error_train_i=sqrt(err_sum);
%calculate the testing error using RSE
deviation=(y_test - (x_test_i * w)).^2;%(200x1 - 200x1=200x1
err_sum=sum(deviation);
error_test_i=sqrt(err_sum);
%%% END PROBLEM 1 CODE %%%


%%% FILL CODE FOR PROBLEM 2 %%%
% ridge regression using least squares - minimization
lambdas = 0:0.00001:0.001;
train_errors = zeros(length(lambdas),1);
test_errors = zeros(length(lambdas),1);
N=size(x_train,1);
for i = 1:length(lambdas)
    lambda = lambdas(i);
    w=learnRidgeRegression(x_train_i,y_train,lambda);%65x1
    % calculate train error using RSE
    err_sum=sum((y_train - (x_train_i*w)).^2);
    train_errors(i,:)=sqrt(err_sum);
    % calculate test error using RSE
    err_sum=sum((y_test - (x_test_i*w)).^2);
    test_errors(i,:)=sqrt(err_sum);
    % fill code here for prediction and computing errors
end

figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');
%%% END PROBLEM 2 CODE %%%



%%% BEGIN PROBLEM 3 CODE
% ridge regression using gradient descent - see handouts (lecture 21 p5) or
% http://cs229.stanford.edu/notes/cs229-notes1.pdf (page 11)
initialWeights = zeros(65,1);
% set the maximum number of iteration in conjugate gradient descent
options = optimset('MaxIter', 500);

% define the objective function
lambdas = 0:0.00001:0.001;
train_errors = zeros(length(lambdas),1);
test_errors = zeros(length(lambdas),1);

% run ridge regression training with fmincg
for i = 1:length(lambdas)
    lambda = lambdas(i);
    objFunction = @(params) regressionObjVal(params, x_train_i, y_train, lambda);
    w = fmincg(objFunction, initialWeights, options);%150. 12.26
    %train error
    deviation=(y_train - (x_train_i*w)).^2;%(242x1 - 242x1=242x1)
    err_sum=sum(deviation);
    train_errors(i,:)=sqrt(err_sum);
    %test error
    deviation=(y_test - (x_test_i*w)).^2;%(242x1 - 242x1=242x1)
    err_sum=sum(deviation);
    test_errors(i,:)=sqrt(err_sum);
    % fill code here for prediction and computing errors
end

figure;
plot([train_errors test_errors]);%732,737    861,800,786
legend('Training Error','Testing Error');
%%% END PROBLEM 3 CODE




%%% BEGIN  PROBLEM 4 CODE
% using variable number 3 only
x_train = x_train(:,3);
x_test = x_test(:,3);
train_errors = zeros(7,1);
test_errors = zeros(7,1);

% no regularization
lambda = 0;
for d = 0:6
    x_train_n = mapNonLinear(x_train,d);
    x_test_n = mapNonLinear(x_test,d);
    w = learnRidgeRegression(x_train_n,y_train,lambda);
    % fill code here for prediction and computing errors
    
    err_sum=sum((y_train - (x_train_n*w)).^2);
    train_errors(d+1,:)=sqrt(err_sum);
    
    err_sum=sum((y_test - (x_test_n*w)).^2);
    test_errors(d+1,:)=sqrt(err_sum);
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');

% optimal regularization
lambda = .001; % from part 2
for d = 0:6
    x_train_n = mapNonLinear(x_train,d);
    x_test_n = mapNonLinear(x_test,d);
    w = learnRidgeRegression(x_train_n,y_train,lambda);
    % fill code here for prediction and computing errors
    
    err_sum=sum((y_train - (x_train_n*w)).^2);
    train_errors(d+1,:)=sqrt(err_sum);
    
    err_sum=sum((y_test - (x_test_n*w)).^2);
    test_errors(d+1,:)=sqrt(err_sum);
end
figure;
plot([train_errors test_errors]);
legend('Training Error','Testing Error');