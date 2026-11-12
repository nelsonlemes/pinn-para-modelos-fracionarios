%% AJUSTE NÃO LINEAR COM FUNÇÃO MITTAG-LEFFLER
clear; clc; close all;

%% Dados experimentais
t = 0:30:2700;

%A = load('dadosC1-120.txt');
%A = load('dadosC1-70.txt');
A = load('dadosC2-70.txt');
t1 = A(:,1);
x1 = A(:,2);

u_ext2 = interp1(t1, x1, t, 'pchip')';  % dados experimentais
x_data = t';
N = length(x_data);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Parâmetros físicos verdadeiros (para gerar dados)
% % M1
% ueq_true1 = 0.082; b_true1 = 0.0030; 
% % M2
% %ueq_true2 = 0.091; b_true2 = 0.010; alpha_true2 = 0.78; 
% 
% % Geração de dados (comportamento de Langmuir clássico + ruído)
% %tau_true2=1;
% u_ext1 = ueq_true1*(1 - exp(-b_true1*x_data));
% %u_ext2 = ueq_true2*(1 - mlf(alpha_true2,1,-b_true2*tau_true2*(x_data/tau_true2).^alpha_true2,12));
% 
% epsilon=0.0;
% 
% % ESCOLHER MODELO
% %MODELO=1;
% u_exp = u_ext1 + epsilon*randn(N,1);
% %u_exp = u_ext2 + epsilon*randn(N,1);
% u_ext2 = u_exp;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Modelo: p = [ueq, b, tau, alpha]
tau=1;
model = @(p,x) p(1)*(1 - mlf(p(3), 1, -p(2)*tau*(x/tau).^p(3), 12));

%% Estimativas iniciais
p0 = [0.07, 0.001, 0.8];

%% Limites: ka, kd, umax > 0; 0 < alpha <= 1
% Altere o limite superior de alpha caso seu modelo admita alpha > 1.
lb = [0.05, eps, 0.01];
ub = [0.1, 1, 0.99];

%% Ajuste por mínimos quadrados não lineares
options = optimoptions('lsqcurvefit', ...
    'Display','iter', ...
    'MaxFunctionEvaluations',1e4, ...
    'MaxIterations',1e3);

[p_fit,resnorm,residual,exitflag,output] = ...
    lsqcurvefit(model, p0, x_data, u_ext2, lb, ub, options);

%% Parâmetros ajustados
ueq_fit    = p_fit(1);
b_fit      = p_fit(2);
alpha_fit  = p_fit(3);

fprintf('\n============================================\n');
fprintf('       RESULTADOS DO AJUSTE\n');
fprintf('============================================\n');
fprintf('ueq    = %.8g\n', ueq_fit);
fprintf('b    = %.8g\n', b_fit);
fprintf('alpha = %.8g\n', alpha_fit);

%% Curva ajustada
x_fit = linspace(min(x_data), max(x_data), 500)';
u_fit = model(p_fit, x_fit);

%% R² e SSE
u_pred = model(p_fit, x_data);

SSE = sum((u_ext2 - u_pred).^2);
SST = sum((u_ext2 - mean(u_ext2)).^2);
R2 = 1 - SSE/SST;

fprintf('R²  = %.8f\n', R2);
fprintf('SSE = %.8e\n', SSE);
fprintf('============================================\n');

%% Gráfico do ajuste
figure;

plot(x_data, u_ext2, 'ko', ...
    'MarkerFaceColor', 'k', ...
    'DisplayName', 'Dados experimentais');
hold on;

plot(x_fit, u_fit, '-', ...
    'LineWidth', 2, ...
    'DisplayName', 'Ajuste não linear');

xlabel('$x$', 'Interpreter', 'latex');
ylabel('$u_{\mathrm{ext2}}$', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
title(sprintf('$R^2 = %.6f$', R2), 'Interpreter', 'latex');
grid on; box on;

