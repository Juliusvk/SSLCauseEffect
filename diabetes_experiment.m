close all; clc; clear all;
T = readtable('diabetes.csv');
A = table2array(T);
Y = A(:,9)+1;
X = A(:,1:8);

[B,dev,stats] = mnrfit(X,Y)

