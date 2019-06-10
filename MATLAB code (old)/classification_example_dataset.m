%% LINEAR EXAMPLE
close all; clc; clear all;

% number of source and target samples
n_S = 2^3; n_T = 2^8;

% conditional for Y given X_C - logistic( sigma_Y * ( x - mu_Y ) )
mu_Y = 0; sigma_Y = 0.5; 
% multimodal input distribution over X_C
mean_mog = 3;
mean_left_S = -mean_mog;
mean_right_S = mean_mog;
std_S = .5;
gm_S = gmdistribution([mean_left_S;mean_right_S],[std_S]);

mean_left_T = -mean_mog;
mean_right_T = mean_mog;
std_T = .5;
gm_T = gmdistribution([mean_left_T;mean_right_T],[std_T]);
X_CS = random(gm_S,n_S)';
X_CT = random(gm_T,n_T)';

% Draw Y (binary classification with logistic conditional Y|X_C):
p_YS = 1./( 1 + exp(-sigma_Y * (X_CS - mu_Y)) ); %P(Y=1|X_CS)
p_YT = 1./( 1 + exp(-sigma_Y * (X_CT - mu_Y)) ); %P(Y=1|X_CT)
Y_S = double(p_YS > rand(1,n_S)); Y_T = double(p_YT > rand(1,n_T));

% Draw X_E (Normally distributed with class-dependent mean and std.):
% conditional for X_E given Y
mu_0 = 0; sigma_0 = .5;
mu_1 = 0; sigma_1 = .25;
a_0 = .5; a_1 = -a_0;

X_ES = zeros(1,n_S); X_ET = zeros(1,n_T);
r_S = randn(1,n_S); r_T = randn(1,n_T);
for i = 1 : n_S
    if Y_S(1,i) == 0
        X_ES(1,i) = mu_0 + sigma_0 * r_S(1,i) + a_0*X_CS(1,i);
    elseif Y_S(1,i) == 1
        X_ES(1,i) = mu_1 + sigma_1 * r_S(1,i) + a_1*X_CS(1,i);
    end
end
for i = 1 : n_T
    if Y_T(1,i) == 0
        X_ET(1,i) = mu_0 + sigma_0 * r_T(1,i) + a_0*X_CT(1,i);
    elseif Y_T(1,i) == 1
        X_ET(1,i) = mu_1 + sigma_1 * r_T(1,i) + a_1*X_CT(1,i);
    end
end


linear = figure(1); hold on;
x_range = linspace(round(min([X_CS,X_CT]))-2,round(max([X_CS,X_CT]))+1,20);
plot(X_CT, X_ET, 'k.', 'LineWidth', 2);
plot(X_CS(Y_S==0), X_ES(Y_S==0), 'rx', 'LineWidth', 2);
plot(X_CS(Y_S==1), X_ES(Y_S==1), 'b^', 'LineWidth', 2);
plot(x_range,mu_0+a_0*x_range,'r:', 'LineWidth',2);
plot(x_range,mu_1+a_1*x_range,'b--', 'LineWidth',2);
hold off;xlabel('X_C');ylabel('X_E'); ylim([-4,4]);
legend('unlabelled','Y=0','Y=1','f_0(X_C)','f_1(X_C)')

%
decision_boundary = figure(2); ll = -2; ul = -ll; step = 0.01;
[X_C,X_E] = meshgrid(ll:step:ul, ll:step:ul);
p_y = 1./(1 + exp(-sigma_Y * (X_C - mu_Y)));
p_e0 = normpdf(X_E, mu_0+a_0 * X_C, sigma_0);
p_e1 = normpdf(X_E, mu_1+a_1 * X_C, sigma_1);
p = p_y.*p_e1./(p_y.*p_e1 + (1-p_y).*p_e0);
imagesc(p); colorbar;


%% NON-LINEAR EXAMPLE
clear all; clc; close all;
x_range = -2:0.01:2;
f0 = @(x) 3 * tanh(x) - cos(1 * x);
f1 = @(x) 5 + 4 * tanh(x) + sin(2 * x);

% number of source and target samples
n_S = 2^3; n_T = 2^8;

% conditional for Y given X_C - logistic( sigma_Y * ( x - mu_Y ) )
mu_Y = 0; sigma_Y = 1; 

% multimodal input distribution over X_C
mean_mog = 1; std_S = .1; 

mean_left_S = -mean_mog; mean_right_S = mean_mog;
gm_S = gmdistribution([mean_left_S;mean_right_S],[std_S]);
mean_left_T = -mean_mog; mean_right_T = mean_mog; std_T = std_S;
gm_T = gmdistribution([mean_left_T;mean_right_T],[std_T]);
X_CS = random(gm_S,n_S)'; X_CT = random(gm_T,n_T)';

% Draw Y (binary classification with logistic conditional Y|X_C):
p_YS = 1./( 1 + exp(-sigma_Y * (X_CS - mu_Y)) ); %P(Y=1|X_CS)
p_YT = 1./( 1 + exp(-sigma_Y * (X_CT - mu_Y)) ); %P(Y=1|X_CT)
Y_S = double(p_YS > rand(1,n_S)); Y_T = double(p_YT > rand(1,n_T));

% Draw X_E (Normally distributed with class-dependent mean and std.):
% conditional for X_E given Y
sigma_0 = 3; sigma_1 = 3;
X_ES = zeros(1,n_S); X_ET = zeros(1,n_T);
r_S = randn(1,n_S); r_T = randn(1,n_T);
for i = 1 : n_S
    if Y_S(1,i) == 0
        X_ES(1,i) = f0(X_CS(1,i)) + sigma_0 * r_S(1,i);
    elseif Y_S(1,i) == 1
        X_ES(1,i) = f1(X_CS(1,i)) + sigma_1 * r_S(1,i);
    end
end
for i = 1 : n_T
    if Y_T(1,i) == 0
        X_ET(1,i) = f0(X_CT(1,i)) + sigma_0 * r_T(1,i);
    elseif Y_T(1,i) == 1
        X_ET(1,i) = f1(X_CT(1,i)) + sigma_1 * r_T(1,i);
    end
end


non_linear = figure(3); hold on;
x_range = linspace(round(min([X_CS,X_CT]))-2,round(max([X_CS,X_CT]))+1,200);
y_range = linspace(round(min([X_ES,X_ET]))-2,round(max([X_ES,X_ET]))+1,200);
plot(X_CT, X_ET, 'k.', 'MarkerSize', 4);
plot(X_CS(Y_S==0), X_ES(Y_S==0), 'ro', 'MarkerSize', 8);
plot(X_CS(Y_S==1), X_ES(Y_S==1), 'b^', 'MarkerSize', 8);
plot(x_range,f0(x_range),'r-', 'LineWidth',2);
plot(x_range,f1(x_range),'b--', 'LineWidth',2);
hold off;xlabel('X_C');ylabel('X_E');
legend('unlabelled','Y=0','Y=1','f_0(X_C)','f_1(X_C)')

%
nonlin_decision_boundary = figure(4);
[X_C,X_E] = meshgrid(x_range, y_range);
p_y = 1./(1 + exp(-sigma_Y * (X_C - mu_Y)));
p_e0 = normpdf(X_E, f0(X_C), sigma_0);
p_e1 = normpdf(X_E, f1(X_C), sigma_1);
p = p_y.*p_e1./(p_y.*p_e1 + (1-p_y).*p_e0);
imagesc(p'); colorbar;


