%% =========================================================
% COMPARAÇÃO DE MODELOS: EDO Clássica (Langmuir) vs EDF Caputo
% LAB QUÍMICA-MATEMÁTICA - NELSON - 2026
%% =========================================================
clear; clc; close all;
rng(42,'twister');

%% 1. GERAÇÃO DOS DADOS SIMULADOS
N = 30;    
TLIMITE = 2700; 
x_data = linspace(0.1, TLIMITE, N)'; 

% Parâmetros físicos do Modelo 1 (EDO Clássica - Langmuir)
ueq_true1 = 0.082; 
b_true1   = 0.0030; 

% Parâmetros físicos do Modelo 2 (EDF Caputo - Mittag-Leffler)
ueq_true2   = 0.086; 
b_true2     = 0.012; 
alpha_true2 = 0.78; 
tau_true2   = 1;

% Soluções nos pontos discretos
u_ext1 = ueq_true1 * (1 - exp(-b_true1 * x_data));
u_ext2 = ueq_true2 * (1 - calcular_mlf(alpha_true2, 1, -b_true2 * tau_true2 * (x_data / tau_true2).^alpha_true2, 12));

% Configuração de ruído e seleção do modelo gerador de dados
epsilon = 7*0.001 / 10;  
MODELO = 1; % 1: EDO Langmuir | 2: EDF Caputo

if MODELO == 2
    u_exp = u_ext1 + epsilon * randn(N, 1);
else
    u_exp = u_ext2 + epsilon * randn(N, 1);
end

%% 2. CURVAS CONTÍNUAS PARA O GRÁFICO (Alta Resolução)
t_fine = linspace(0.1, TLIMITE, 500)';

% Avaliação contínua do Modelo 1
u_fine1 = ueq_true1 * (1 - exp(-b_true1 * t_fine));

% Avaliação contínua do Modelo 2
u_fine2 = ueq_true2 * (1 - calcular_mlf(alpha_true2, 1, -b_true2 * tau_true2 * (t_fine / tau_true2).^alpha_true2, 12));

%% 3. CONSTRUÇÃO E FORMATACÃO DO GRÁFICO
figure('Color', 'w', 'Position', [150, 150, 850, 520]);

% Plot 1: Dados discretos (Simulados com ruído)
plot(x_data, u_exp, 'ko', 'LineWidth', 1.2, 'MarkerSize', 6, ...
     'MarkerFaceColor', [0.2 0.2 0.2], ...
     'DisplayName', sprintf('Dados Simulados (Gerador: Modelo %d)', MODELO));
hold on;

% Plot 2: Modelo 1 (EDO Clássica)
plot(t_fine, u_fine1, 'b--', 'LineWidth', 2.0, ...
     'DisplayName', 'Modelo 1: EDO Clássica (Langmuir)');

% Plot 3: Modelo 2 (EDF Caputo)
plot(t_fine, u_fine2, 'r-', 'LineWidth', 2.0, ...
     'DisplayName', sprintf('Modelo 2: EDF Caputo (\\alpha = %.2f)', alpha_true2));

% Ajustes estéticos
xlabel('Tempo t (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('u(t)', 'FontSize', 12, 'FontWeight', 'bold');
title('Comparação de Modelos: EDO Clássica vs. EDF de Caputo', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 10);
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5, 'FontSize', 11);
xlim([0, TLIMITE]);
ylim([0, max([u_fine1; u_fine2]) * 1.1]);

hold off;

%% FUNÇÃO AUXILIAR PARA A FUNÇÃO DE MITTAG-LEFFLER
function E = calcular_mlf(alpha, beta, z, p)
    % Utiliza a função mlf.m nativa se existir no diretório, 
    % caso contrário utiliza aproximação por série de Taylor/Gamma.
    if exist('mlf', 'file') == 2 || exist('mlf', 'builtin') == 5
        E = mlf(alpha, beta, z, p);
    else
        E = zeros(size(z));
        for i = 1:numel(z)
            zi = z(i);
            s = 0;
            for k = 0:100
                term = (zi^k) / gamma(alpha * k + beta);
                s = s + term;
                if abs(term) < 1e-14 && k > 15
                    break;
                end
            end
            E(i) = s;
        end
    end
end