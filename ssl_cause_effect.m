%% This script generates data for SSL with cause and effect features in the
%% general case, i.e. with the link C -> E allowed
close all;

%% First we consider the identifiable case with linear Gaussian structural 
%% equations with equal error variances

% first define the parameters
d_c = 1; %dim. of X_C
d_e = 1; %dim of X_E
n_l = 3; %no. of labelled data
n_u = 20; %no. of unlabelled data
sigma = 2; %error std. dev. (equal for all of X_C,Y,X_E)
a_CY = 2; %lin. coeff. vector from C to Y
a_CE = 3; %lin. coeff. vector from C to E
a_YE = 2; %lin. coeff. from Y to E

% generate labelled sample
C = sigma * randn(n_l, d_c); %draw C from N(0,sigma^2)
Y = a_CY * C + sigma * randn(n_l, 1); %draw Y from N(a_CY*C, sigma^2)
E = a_CE * C + a_YE * Y + sigma * randn(n_l, 1);

% regress Y on C
% C_1 = [C, ones(n_l,1)];
% b_Y1 = (C_1'*C_1)\C_1'*Y
b_CY = (C'*C)\C'*Y;

% regress E on Y and C
% CY_1 = [C, Y, ones(n_l,1)];
% b_E1 =(CY_1'*CY_1)\CY_1'*E
b_E = ([C,Y]'*[C,Y])\[C,Y]'*E;
b_CE = b_E(1);
b_YE = b_E(2);

% generate unlabelled sample
C_u = -3 + sigma * randn(n_u, d_c); %draw C from N(0,sigma^2)
Y_u = a_CY * C_u + sigma * randn(n_u, 1); %draw Y from N(a_CY*C, sigma^2)
E_u = a_CE * C_u + a_YE * Y_u + sigma * randn(n_u, 1);

%% plot fitted curve for C vs E
xvals = linspace(-5,5,3);
plot(C,E,'bx')
hold on
plot(C_u,E_u,'ro')
plot(xvals,(b_CE + b_CY * b_YE)*xvals,'b')
plot(xvals,(a_CE + a_CY * a_YE)*xvals,'r')
xlabel('Causes X_C')
ylabel('Effects X_E')
legend('Labelled data', 'Unlabelled data', 'Fit from labelled data', 'True Model')

