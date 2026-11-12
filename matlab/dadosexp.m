% Script para reproduzir curvas de c/c0 (Fig. 7) - C1-70 e C1-120
clear all; clc;

% Tempo (minutos) conforme escala da Fig. 7
t = 0:1:2800; 

%% 1. Parâmetros Multi-exponenciais (Tabela 3) - c_eq/c0 = 0
% Amostra C1-70
f_mexp_C1_70 = [0.082, 0.121, 0.797]; 
k_mexp_C1_70 = [0.70, 0.016, 0.0022]; aeq_C1_70 = 0.09304;
F = f_mexp_C1_70(1)*(1-exp(-k_mexp_C1_70(1)*t)) + ...
                  f_mexp_C1_70(2)*(1-exp(-k_mexp_C1_70(2)*t)) + ...
                  f_mexp_C1_70(3)*(1-exp(-k_mexp_C1_70(3)*t));

a_mexp_C1_70=aeq_C1_70*F;

% Amostra C1-120
f_mexp_C1_120 = [0.037, 0.919, 0.044]; 
k_mexp_C1_120 = [4.52, 0.004, 0.0009]; aeq_C1_120 = 0.086;
F = f_mexp_C1_120(1)*(1-exp(-k_mexp_C1_120(1)*t)) + ...
                   f_mexp_C1_120(2)*(1-exp(-k_mexp_C1_120(2)*t)) + ...
                   f_mexp_C1_120(3)*(1-exp(-k_mexp_C1_120(3)*t));


a_mexp_C1_120=aeq_C1_120 *F;

%% 2. Parâmetros de Ordem Mista MOE (Tabela 4)
% Amostra C1-70 (ceq/c0 = 0)
f2_moe_C1_70 = 0.424; k1_moe_C1_70 = 0.00178; aeq_C1_70 = 0.09304;
F=(1 - exp(-k1_moe_C1_70 * t)) ./ ...
                 (1 - f2_moe_C1_70 * exp(-k1_moe_C1_70 * t));
a_moe_C1_70 = aeq_C1_70 * F;

% Amostra C1-120 (ceq/c0 = 0.00879)
f2_moe_C1_120 = 0.095; k1_moe_C1_120 = 0.00352; aeq_C1_120 = 0.086;
F=(1 - exp(-k1_moe_C1_120 * t)) ./ ...
                  (1 - f2_moe_C1_120 * exp(-k1_moe_C1_120 * t));
a_moe_C1_120 = aeq_C1_120 * F;

%% 3. Gráficos de Concentração Relativa (c/c0)
figure('Color', 'w');

%% Dados experimentais
A = load('dadosC1-70.txt');
t1 = A(:,1);
x1 = A(:,2);
B = load('dadosC1-120.txt');
t2 = B(:,1);
x2 = B(:,2);

% Subplot para C1-70
subplot(1,2,1);
loglog(t, a_mexp_C1_70, 'b-', 'LineWidth', 2); hold on;
loglog(t, a_moe_C1_70, 'r--', 'LineWidth', 2);
loglog(t1,x1,'bo');
title('Progresso de Adsorção: C1-70');
xlabel('Tempo (min)'); ylabel('a (mmol/g)');
ylim([0 0.105]); grid on;
legend('m-exp', 'MOE', 'EXP', 'Location', 'southeast');

% Subplot para C1-120
subplot(1,2,2);
loglog(t, a_mexp_C1_120, 'b-', 'LineWidth', 2); hold on;
loglog(t, a_moe_C1_120, 'r--', 'LineWidth', 2);
loglog(t2,x2,'bo');
title('Progresso de Adsorção: C1-120');
xlabel('Tempo (min)'); ylabel('a (mmol/g)');
ylim([0 0.105]); grid on;
legend('m-exp', 'MOE', 'EXP', 'Location', 'southeast');



