%% information
% facial age estimation
% regression method: linear regression
% last updated: Feb 2013

%% settings
clear;
clc;

% path 
database_path = './dataset/data_age.mat';
result_path = './results/';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
err_level = 5;

%% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

yhat_test = xtest * w_lr;

%% Compute the MAE and CS value (with cumulative error level of 5) for linear regression 
mae_lr = sum(abs(yhat_test-ytest))/size(ytest, 1);
cum_err_lr = cal_cum_err(yhat_test, ytest);
fprintf('MAE(Linear regression) = %f\n', mae_lr);
fprintf('CS(5) = %f\n', cum_err_lr(5));
%% generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15. The plot should look at the one in the Week6 lecture slides
plot(1:15, cum_err_lr(1:15))
%% Compute the MAE and CS value (with cumulative error level of 5) for both partial least square regression and the regression tree model by using the Matlab built in functions.
% Partial least square regression
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(xtrain, ytrain);
yhat_test_plsr = [ones(size(xtest,1),1) xtest]*beta;
cum_err_plsr = cal_cum_err(yhat_test_plsr, ytest);
mae_plsr = sum(abs(yhat_test_plsr-ytest))/size(ytest, 1);
fprintf('MAE(partial least square regression) = %f\n', mae_plsr);
fprintf('CS(5) = %f\n', cum_err_plsr(5));

% Regression tree
w_rt_tree = fitrtree(xtrain, ytrain);
yhat_test_rt = predict(w_rt_tree, xtest);
cum_err_rt = cal_cum_err(yhat_test_rt, ytest);
mae_rt = sum(abs(yhat_test_rt-ytest))/size(ytest, 1);
fprintf('MAE(regression tree) = %f\n', mae_rt);
fprintf('CS(5) = %f\n', cum_err_rt(5));

%% Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regression by using LIBSVM toolbox
addpath(genpath('software'));
st = svmtrain(ytrain, xtrain, '-s 3 -t 0');
yhat_test_svm = svmpredict(ytest, xtest, st);
cum_err_svm = cal_cum_err(yhat_test_svm, ytest);
mae_svm = sum(abs(yhat_test_svm-ytest))/size(ytest, 1);
fprintf('MAE(SVR) = %f\n', mae_svm);
fprintf('CS(5) = %f\n', cum_err_svm(5));

%% Plot


plot(1:15, cum_err_lr(1:15), 'g-o'); hold on;
plot(1:15, cum_err_plsr(1:15) ,'r-o'); hold on;
plot(1:15, cum_err_rt(1:15), 'black-o'); hold on;
plot(1:15, cum_err_svm(1:15),'b-o'); hold off;
grid on
title('CS plot')
legend('Linear regression','Partial least square regression','Regression tree','SVR')
legend('Location','southeast')
ylabel('Cummulative score')
xlabel('Error levels (year)')
