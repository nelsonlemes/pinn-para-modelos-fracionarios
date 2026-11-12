% =========================================================================
% SCRIPT: Visualização das Funções de Perda (Loss History) - PINNs
% LAB QUÍMICA-MATEMÁTICA - UNIFAL-MG
% =========================================================================

% 1. Verificação de existência das variáveis no Workspace
vars = {'p11', 'p21', 'p12', 'p22', 'p32'};
for k = 1:length(vars)
    if ~exist(vars{k}, 'var')
        error('A variável "%s" não foi encontrada no Workspace.', vars{k});
    end
end

% 2. Conversão de dlarray para vetores numéricos do MATLAB
p11_num = double(extractdata(p11));
p21_num = double(extractdata(p21));

p12_num = double(extractdata(p12));
p22_num = double(extractdata(p22));
p32_num = double(extractdata(p32));

% 3. Suavização das curvas de perda via Média Móvel (movmean)
janela = 200; % Número de épocas para a janela de suavização (ajustável)

p11_suave = movmean(p11_num, janela);
p21_suave = movmean(p21_num, janela);

p12_suave = movmean(p12_num, janela);
p22_suave = movmean(p22_num, janela);
p32_suave = movmean(p32_num, janela);

% 4. Preparação do vetor de iterações/épocas
N_iter = length(p11_num);
iter = (1:N_iter)'; % Vetor coluna

% 5. Configuração e Plotagem da Figura
figure(3)
%figure('Color', 'w', 'Position', [100, 100, 1100, 480]);

% -------------------------------------------------------------------------
% PAINEL 1: Modelo 1 (EDO Clássica - M1)
% -------------------------------------------------------------------------
%subplot(1, 2, 1);
%semilogy(iter, 10*E1a_suave, 'k-', 'LineWidth', 2, 'DisplayName', 'Loss Total (\mathcal{L}_1)'); hold on;
plot(iter, p11_suave, 'b--', 'LineWidth', 1, 'DisplayName', 'M1'); hold on;
plot(iter, p12_suave, 'r:', 'LineWidth', 1.5, 'DisplayName', 'M2');
hold off;

grid on; box on;
xlabel('Iterações / Épocas', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('u_eq', 'FontSize', 11, 'FontWeight', 'bold');
%title('Evolução dos parametros', 'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 10, 'Interpreter', 'tex');
%ylim([min([E1a_suave(:); E2a_suave(:); E3a_suave(:)])*0.5, max([E1a_suave(:); E2a_suave(:); E3a_suave(:)])*2]);

% -------------------------------------------------------------------------
% PAINEL 2: Modelo 2 (EDF Caputo - M2)
% -------------------------------------------------------------------------
figure(4)
%subplot(1, 2, 2);
%semilogy(iter, 10*E1b_suave, 'k-', 'LineWidth', 2, 'DisplayName', 'Loss Total (\mathcal{L}_2)'); hold on;
plot(iter, p21_suave, 'b--', 'LineWidth', 1, 'DisplayName', 'M1'); hold on;
plot(iter, p22_suave, 'r:', 'LineWidth', 1.5, 'DisplayName', 'M2');
hold off;

grid on; box on;
xlabel('Iterações / Épocas', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('b', 'FontSize', 11, 'FontWeight', 'bold');
%title('Evolução das parametros', 'FontSize', 12);
legend('Location', 'northeast', 'FontSize', 10, 'Interpreter', 'tex');
%ylim([min([E1b_suave(:); E2b_suave(:); E3b_suave(:)])*0.5, max([E1b_suave(:); E2b_suave(:); E3b_suave(:)])*2]);

%subplot(1, 3, 3);
figure(5)
%semilogy(iter, 10*E1b_suave, 'k-', 'LineWidth', 2, 'DisplayName', 'Loss Total (\mathcal{L}_2)'); hold on;
plot(iter, p32_suave, 'b--', 'LineWidth', 1, 'DisplayName', 'Loss Dados (\mathcal{L}_{data})'); 
% hold on;
% hold off;

grid on; box on;
xlabel('Iterações / Épocas', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('\\alphha', 'FontSize', 11, 'FontWeight', 'bold');
%title('Evolução das parametros - Modelo 2 (\mathcal{M}_2: EDF)', 'FontSize', 12);
%legend('Location', 'northeast', 'FontSize', 10, 'Interpreter', 'tex');



% %sgtitle('Histórico de Convergência do Treinamento das PINNs (Suavizado)', 'FontSize', 14, 'FontWeight', 'bold');
% 
% % =========================================================================
% % 6. SALVAR OS DADOS EM UM ARQUIVO DE TEXTO (.TXT)
% % =========================================================================
% 
% % Garantir que todos os vetores estejam no formato de coluna (N x 1)
% E1a_col = E1a_num(:); E2a_col = E2a_num(:); E3a_col = E3a_num(:);
% E1b_col = E1b_num(:); E2b_col = E2b_num(:); E3b_col = E3b_num(:);
% 
% E1a_s_col = E1a_suave(:); E2a_s_col = E2a_suave(:); E3a_s_col = E3a_suave(:);
% E1b_s_col = E1b_suave(:); E2b_s_col = E2b_suave(:); E3b_s_col = E3b_suave(:);
% 
% % Criação da Tabela com colunas identificadas
% T_perdas = table(iter, ...
%     E1a_col, E2a_col, E3a_col, ...
%     E1b_col, E2b_col, E3b_col, ...
%     E1a_s_col, E2a_s_col, E3a_s_col, ...
%     E1b_s_col, E2b_s_col, E3b_s_col, ...
%     'VariableNames', { ...
%         'Epoca', ...
%         'M1_LossTotal_Bruta', 'M1_LossData_Bruta', 'M1_LossPhys_Bruta', ...
%         'M2_LossTotal_Bruta', 'M2_LossData_Bruta', 'M2_LossPhys_Bruta', ...
%         'M1_LossTotal_Suave', 'M1_LossData_Suave', 'M1_LossPhys_Suave', ...
%         'M2_LossTotal_Suave', 'M2_LossData_Suave', 'M2_LossPhys_Suave' ...
%     });
% 
% % Nome do arquivo de saída
% nome_arquivo = 'historico_perdas_pinns2.txt';
% 
% % Escrever os dados no arquivo TXT delimitado por tabulação
% writetable(T_perdas, nome_arquivo, 'Delimiter', '\t');
% 
% fprintf('\n---------------------------------------------------------\n');
% fprintf('Dados de perda salvos com sucesso no arquivo: %s\n', nome_arquivo);
% fprintf('---------------------------------------------------------\n');