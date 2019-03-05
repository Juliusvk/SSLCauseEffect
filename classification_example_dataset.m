close all; clc;

% number of source and target samples
n_S = 2^3; n_T = 2^8;

% source and target marginals for X_C - Normal distributions
mu_S = 0; sigma_S = 1;
mu_T = 0; sigma_T = 2;

% conditional for Y given X_C - logistic( sigma_Y * ( x - mu_Y ) )
mu_Y = 0; sigma_Y = 1; 

% conditional for X_E given Y
mu_0 = -10; sigma_0 = 2;
mu_1 = 10; sigma_1 = 2;
a = 1;

% Draw X_C (Normally distributed with domain-dependent mean and std.):
% X_CS = mu_S + sigma_S * randn(1,n_S);
% X_CT = mu_T + sigma_T * randn(1,n_T);

% multimodal input distribution over X_C
mean_left_S = 0;
mean_right_S = 4;
std_S = .5;
gm_S = gmdistribution([mean_left_S;mean_right_S],[std_S]);

mean_left_T = -4;
mean_right_T = 0;
std_T = .5;
gm_T = gmdistribution([mean_left_T;mean_right_T],[std_T]);

X_CS = random(gm_S,n_S)';
X_CT = random(gm_T,n_T)';



% Draw Y (binary classification with logistic conditional Y|X_C):
p_YS = 1./( 1 + exp(-sigma_Y * (X_CS - mu_Y)) ); %P(Y=1|X_CS)
p_YT = 1./( 1 + exp(-sigma_Y * (X_CT - mu_Y)) ); %P(Y=1|X_CT)
Y_S = double(p_YS > rand(1,n_S)); Y_T = double(p_YT > rand(1,n_T));

% Draw X_E (Normally distributed with class-dependent mean and std.):
X_ES = zeros(1,n_S); X_ET = zeros(1,n_T);

r_S = randn(1,n_S); r_T = randn(1,n_T);
for i = 1 : n_S
    if Y_S(1,i) == 0
        X_ES(1,i) = mu_0 + sigma_0 * r_S(1,i) + a*X_CS(1,i)^2;
    elseif Y_S(1,i) == 1
        X_ES(1,i) = mu_1 + sigma_1 * r_S(1,i) + a*X_CS(1,i)^2;
    end
end
for i = 1 : n_T
    if Y_T(1,i) == 0
        X_ET(1,i) = mu_0 + sigma_0 * r_T(1,i) + a*X_CT(1,i).^2;
    elseif Y_T(1,i) == 1
        X_ET(1,i) = mu_1 + sigma_1 * r_T(1,i) + a*X_CT(1,i).^2;
    end
end

D_S = [X_CS; Y_S; X_ES];
D_T = [X_CT; Y_T; X_ET];

x_range = linspace(round(min([X_CS,X_CT]))-1,round(max([X_CS,X_CT]))+1,20);

figure(2);
    hold on
    plot(X_CT(Y_T==0), X_ET(Y_T==0), 'k+', 'LineWidth',.5)
    plot(X_CT(Y_T==1), X_ET(Y_T==1), 'ks', 'LineWidth',.5)
    plot(X_CS(Y_S==0), X_ES(Y_S==0), 'rx', 'LineWidth',2)
    plot(X_CS(Y_S==1), X_ES(Y_S==1), 'bo', 'LineWidth',2)
    plot(x_range,mu_0+a*x_range.^2,'r-')
    plot(x_range,mu_1+a*x_range.^2,'b-')
    hold off
    xlabel('X_C')
    ylabel('X_E')
    legend('Y=0 (unlabelled)','Y=1 (unlabelled)',...
        'Y=0 (labelled)','Y=1 (labelled)','f_0(X_C)','f_1(X_C)')
