% =========================================================================
% SCRIPT: Visualização das Funções de Perda (Loss History) - PINNs
% LAB QUÍMICA-MATEMÁTICA - UNIFAL-MG
% =========================================================================

% 1. Verificação de existência das variáveis no Workspace
vars = {'E1a', 'E2a', 'E3a', 'E1b', 'E2b', 'E3b'};
for k = 1:length(vars)
    if ~exist(vars{k}, 'var')
        error('A variável "%s" não foi encontrada no Workspace.', vars{k});
    end
end

% 2. Conversão de dlarray para vetores numéricos do MATLAB
E1a_num = double(extractdata(E1a));
E2a_num = double(extractdata(E2a));
E3a_num = double(extractdata(E3a));

E1b_num = double(extractdata(E1b));
E2b_num = double(extractdata(E2b));
E3b_num = double(extractdata(E3b));

% 3. Suavização das curvas de perda via Média Móvel (movmean)
janela = 200; % Número de épocas para a janela de suavização (ajustável)

E1a_suave = movmean(E1a_num, janela);
E2a_suave = movmean(E2a_num, janela);
E3a_suave = movmean(E3a_num, janela);

E1b_suave = movmean(E1b_num, janela);
E2b_suave = movmean(E2b_num, janela);
E3b_suave = movmean(E3b_num, janela);

% 4. Preparação do vetor de iterações/épocas
N_iter = length(E1a_num);
iter = (1:N_iter)'; % Vetor coluna

% 5. Configuração e Plotagem da Figura
figure('Color', 'w', 'Position', [100, 100, 1100, 480]);

% -------------------------------------------------------------------------
% PAINEL 1: Modelo 1 (EDO Clássica - M1)
% -------------------------------------------------------------------------
subplot(1, 2, 1);
%semilogy(iter, 10*E1a_suave, 'k-', 'LineWidth', 2, 'DisplayName', 'Loss Total (\mathcal{L}_1)'); hold on;
semilogy(iter, E2a_suave, 'b--', 'LineWidth', 1, 'DisplayName', 'Loss Dados (\mathcal{L}_{data})'); hold on;
semilogy(iter, E3a_suave, 'r:', 'LineWidth', 1.5, 'DisplayName', 'Loss Física (\mathcal{L}_{phys})');
hold off;

grid on; box on;
xlabel('Iterações / Épocas', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Função de Perda (Escala Log)', 'FontSize', 11, 'FontWeight', 'bold');
title('Evolução das Perdas - Modelo 1 (\mathcal{M}_1: EDO)', 'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 10, 'Interpreter', 'tex');
ylim([min([E1a_suave(:); E2a_suave(:); E3a_suave(:)])*0.5, max([E1a_suave(:); E2a_suave(:); E3a_suave(:)])*2]);

% -------------------------------------------------------------------------
% PAINEL 2: Modelo 2 (EDF Caputo - M2)
% -------------------------------------------------------------------------
subplot(1, 2, 2);
%semilogy(iter, 10*E1b_suave, 'k-', 'LineWidth', 2, 'DisplayName', 'Loss Total (\mathcal{L}_2)'); hold on;
semilogy(iter, E2b_suave, 'b--', 'LineWidth', 1, 'DisplayName', 'Loss Dados (\mathcal{L}_{data})'); hold on;
semilogy(iter, E3b_suave, 'r:', 'LineWidth', 1.5, 'DisplayName', 'Loss Física (\mathcal{L}_{phys})');
hold off;

grid on; box on;
xlabel('Iterações / Épocas', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Função de Perda (Escala Log)', 'FontSize', 11, 'FontWeight', 'bold');
title('Evolução das Perdas - Modelo 2 (\mathcal{M}_2: EDF)', 'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 10, 'Interpreter', 'tex');
ylim([min([E1b_suave(:); E2b_suave(:); E3b_suave(:)])*0.5, max([E1b_suave(:); E2b_suave(:); E3b_suave(:)])*2]);

sgtitle('Histórico de Convergência do Treinamento das PINNs (Suavizado)', 'FontSize', 14, 'FontWeight', 'bold');

% =========================================================================
% 6. SALVAR OS DADOS EM UM ARQUIVO DE TEXTO (.TXT)
% =========================================================================

% Garantir que todos os vetores estejam no formato de coluna (N x 1)
E1a_col = E1a_num(:); E2a_col = E2a_num(:); E3a_col = E3a_num(:);
E1b_col = E1b_num(:); E2b_col = E2b_num(:); E3b_col = E3b_num(:);

E1a_s_col = E1a_suave(:); E2a_s_col = E2a_suave(:); E3a_s_col = E3a_suave(:);
E1b_s_col = E1b_suave(:); E2b_s_col = E2b_suave(:); E3b_s_col = E3b_suave(:);

% Criação da Tabela com colunas identificadas
T_perdas = table(iter, ...
    E1a_col, E2a_col, E3a_col, ...
    E1b_col, E2b_col, E3b_col, ...
    E1a_s_col, E2a_s_col, E3a_s_col, ...
    E1b_s_col, E2b_s_col, E3b_s_col, ...
    'VariableNames', { ...
        'Epoca', ...
        'M1_LossTotal_Bruta', 'M1_LossData_Bruta', 'M1_LossPhys_Bruta', ...
        'M2_LossTotal_Bruta', 'M2_LossData_Bruta', 'M2_LossPhys_Bruta', ...
        'M1_LossTotal_Suave', 'M1_LossData_Suave', 'M1_LossPhys_Suave', ...
        'M2_LossTotal_Suave', 'M2_LossData_Suave', 'M2_LossPhys_Suave' ...
    });

% Nome do arquivo de saída
nome_arquivo = 'historico_perdas_pinns2.txt';

% Escrever os dados no arquivo TXT delimitado por tabulação
writetable(T_perdas, nome_arquivo, 'Delimiter', '\t');

fprintf('\n---------------------------------------------------------\n');
fprintf('Dados de perda salvos com sucesso no arquivo: %s\n', nome_arquivo);
fprintf('---------------------------------------------------------\n');