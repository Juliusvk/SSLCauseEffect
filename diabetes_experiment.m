close all; clc; clear all;

%% load and preprocess data
T = readtable('diabetes.csv'); A = table2array(T);
X = A(:,1:8); Y = A(:,9)+1;

% train and test on full dataset: oracle
[B_full,~,~] = mnrfit(X, Y);
[AUC_full, x_roc_full, y_roc_full] = evaluate(B_full, X, Y);

% Feature selection: Pregnancy, Glucose, BMI, PedigreeFunc. (p < 0.01)
X = A([1, 2, 6, 7],1:8); 

%% separate into train and test data
n_train = 512;
n_test = 256;
[X_tr, Y_tr, X_te, Y_te] = train_test_split(X, Y, n_train, n_test);

%% split into labelled and unlabelled
n_labelled = 64;
[X_lab, Y_lab, X_unl, Y_unl] = train_test_split(X_tr, Y_tr,...
    n_labelled, n_train - n_labelled);

%% Baseline1: logistic regression on labelled data only
[B,~,~] = mnrfit(X_lab, Y_lab);
[AUC_sup, x_roc_sup, y_roc_sup] = evaluate(B_feat, X_te, Y_te);





