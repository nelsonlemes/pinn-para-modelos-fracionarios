%% AJUSTE NÃO LINEAR DOS DADOS EXPERIMENTAIS
clear; clc; close all;

%% ============================================================
% Dados experimentais
% =============================================================

% u_ext1  -> variável experimental
% x_data  -> variável independente

t = 0:30:2700; 

A = load('dadosC1-120.txt');
%A = load('dadosC1-70.txt');
%A = load('dadosC2-70.txt');
t1 = A(:,1);
x1 = A(:,2);
a_C1_70 = interp1(t1, x1, t, 'pchip');

x_data=t'; 
N=length(x_data);
u_ext1=a_C1_70';

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parâmetros físicos verdadeiros (para gerar dados)
% M1
% ueq_true1 = 0.082; b_true1 = 0.0030; 
% M2
% ueq_true2 = 0.091; b_true2 = 0.010; alpha_true2 = 0.78; 
% 
% Geração de dados (comportamento de Langmuir clássico + ruído)
% tau_true2=1;
% u_ext1 = ueq_true1*(1 - exp(-b_true1*x_data));
% u_ext2 = ueq_true2*(1 - mlf(alpha_true2,1,-b_true2*tau_true2*(x_data/tau_true2).^alpha_true2,12));
% 
% epsilon=0.0;
% 
% ESCOLHER MODELO
% MODELO=1;
% u_exp = u_ext1 + epsilon*randn(N,1);
% u_exp = u_ext2 + epsilon*randn(N,1);
% u_ext2 = u_exp;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ============================================================
% Modelo
% =============================================================

model = @(p,x) p(1)*(1 - exp(-p(2)*x));

% p(1) = ueq
% p(2) = b

%% ============================================================
% Estimativas iniciais
% =============================================================

p0 = [.07, 0.01];

%% ============================================================
% Limites dos parâmetros
% =============================================================

lb = [0.05, eps];
ub = [0.1, 1];

%% ============================================================
% Ajuste por mínimos quadrados não lineares
% =============================================================

options = optimoptions('lsqcurvefit', ...
    'Display','iter', ...
    'MaxFunctionEvaluations',1e4, ...
    'MaxIterations',1e3);

[p_fit,resnorm,residual,exitflag,output] = ...
    lsqcurvefit(model,p0,x_data,u_ext1,lb,ub,options);

%% ============================================================
% Parâmetros ajustados
% =============================================================

ueq_fit = p_fit(1);
b_fit = p_fit(2);

fprintf('\n============================================\n');
fprintf('       RESULTADOS DO AJUSTE\n');
fprintf('============================================\n');

fprintf('ueq = %.8g\n',ueq_fit);
fprintf('b = %.8g\n',b_fit);

%% ============================================================
% Curva ajustada
% =============================================================

x_fit = linspace(min(x_data),max(x_data),500);

u_fit = model(p_fit,x_fit);

%% ============================================================
% Coeficiente de determinação R²
% =============================================================

u_pred = model(p_fit,x_data);

SSE = sum((u_ext1-u_pred).^2);
SST = sum((u_ext1-mean(u_ext1)).^2);

R2 = 1 - SSE/SST;

fprintf('R² = %.8f\n',R2);
fprintf('SSE = %.8e\n',SSE);
fprintf('============================================\n');

%% ============================================================
% Gráfico
% =============================================================

figure;

plot(x_data,u_ext1,'ko', ...
    'MarkerFaceColor','k', ...
    'DisplayName','Dados experimentais');

hold on;

plot(x_fit,u_fit,'-', ...
    'LineWidth',2, ...
    'DisplayName','Ajuste não linear');

xlabel('$x$','Interpreter','latex');
ylabel('$u_{\mathrm{ext1}}$','Interpreter','latex');

legend('Location','best','Interpreter','latex');

grid on;
box on;

title(sprintf('$R^2 = %.6f$',R2), ...
    'Interpreter','latex');

