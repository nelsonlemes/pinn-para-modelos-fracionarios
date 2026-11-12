%% =========================================================
% PINN PARA SELEÇÃO DE MODELOS: EDO Clássica vs EDF Caputo
% Diferenciação Numérica (5 pontos) e Integração Fracionária (L1)
%% =========================================================
clear; clc; close all;
rng(42,'twister');

%% 1. CARREGANDO OS DADOS EXPERIMENTAIS
t = 0.1:8.5:2700; 
h = 8.5; 
EXPDADOS='C2-70';
%A = load('dadosC1-70.txt');
%A = load('dadosC1-120.txt');
A = load('dadosC2-70.txt');
t1 = A(:,1);
x1 = A(:,2);
a_dados = interp1(t1, x1, t, 'pchip');
x_data = t';
u_exp = a_dados';
N = length(x_data);

%% [MODIFICAÇÃO 1]: ADIMENSIONAMENTO DO TEMPO
t_max = max(x_data);
x_data_norm = x_data / t_max; % Tempo normalizado em [0, 1]
h_norm = h / t_max;           % Passo de tempo adimensionado

%% 2. ARQUITETURA DA REDE NEURAL E PARÂMETROS
ARQ=10;
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(ARQ)
    tanhLayer
    fullyConnectedLayer(ARQ)
    tanhLayer
    fullyConnectedLayer(1)
];
net_M1 = dlnetwork(layers); % Rede para o Modelo 1 (EDO)
net_M2 = dlnetwork(layers); % Rede para o Modelo 2 (EDF)

% Faixas dos parâmetros de ajuste
ueq_min = 0.05;   ueq_max = 0.12;
b_min   = 1e-4;  b_max   = 1;
alpha_min = 0.55;  alpha_max = 0.999;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = @(p,x) p(1)*(1 - exp(-p(2)*x));
p0 = [5.00e-02, 3.44e-01];

lb = [ueq_min, b_min]/1000;
ub = [ueq_max, b_max]*100;

options = optimoptions('lsqcurvefit', ...
    'Display','iter', ...
    'MaxFunctionEvaluations',1e4, ...
    'MaxIterations',1e3);

[p_fit,resnorm,residual,exitflag,output] = ...
    lsqcurvefit(model,p0,x_data,u_exp,lb,ub,options);

ueq_chute1  = p_fit(1);
b_chute1  = p_fit(2);
u_chute1 = ueq_chute1*(1 - exp(-b_chute1*x_data));

%% ============================================================
% Modelo
% =============================================================

tau=1;
model = @(p,x) p(1)*(1 - mlf(p(3), 1, -p(2)*tau*(x/tau).^p(3), 16));

%% Estimativas iniciais
p0 = [(ueq_min+ueq_max)/2, (b_min+b_max)/2, 0.55];

%% Limites: ka, kd, umax > 0; 0 < alpha <= 1
% Altere o limite superior de alpha caso seu modelo admita alpha > 1.
lb = [ueq_min/1000, b_min/1000, 0.55];
ub = [ueq_max*100, b_max*100, 0.999];

%% Ajuste por mínimos quadrados não lineares
options = optimoptions('lsqcurvefit', ...
    'Display','iter', ...
    'MaxFunctionEvaluations',1e4, ...
    'MaxIterations',1e3);

[p_fit,resnorm,residual,exitflag,output] = ...
    lsqcurvefit(model, p0, x_data, u_exp, lb, ub, options);

%% Parâmetros ajustados
ueq_chute2    = p_fit(1);
b_chute2      = p_fit(2);
alpha_chute2  = p_fit(3);
tau_chute2 =1;
u_chute2 = ueq_chute2*(1 - mlf(alpha_chute2,1,-b_chute2*tau_chute2*(x_data/tau_chute2).^alpha_chute2,16));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

resx_M1 = u_exp(:) - u_chute1(:);
resx_M2 = u_exp(:) - u_chute2(:);
 
MSEx_M1 = mean(resx_M1.^2);
MSEx_M2 = mean(resx_M2.^2);

% Valores iniciais (dlarrays para rastreamento de gradiente)
% M1
ueq_M1 = dlarray(ueq_chute1); b_M1 = dlarray(b_chute1); 
% M2
ueq_M2 = dlarray(ueq_chute2); b_M2 = dlarray(b_chute2); alpha_M2 = dlarray(alpha_chute2); 

% Parâmetros de Treinamento
numEpochs = 50000;%50000; 
learnRate = 1e-3;
lr_ueq    = 1e-2;
lr_b      = 5e-3;
lr_alpha  = 1e-2;

%% [MODIFICAÇÃO 2]: PESOS DE PERDA REDIMENSIONADOS
lambda_phys = 1000;  % Reduzido para evitar hiper-regularização
lambda_ic   = 1000; % Peso para a condição inicial u(0) = 0

% Otimizador Adam (M1 e M2)
avg_M1 = []; avgSq_M1 = [];
avg_M2 = []; avgSq_M2 = [];

%% 3. LOOP DE TREINAMENTO VARIACIONAL
dlX = dlarray(x_data_norm', 'CB'); % Entradas com tempo normalizado
dlU = dlarray(u_exp', 'CB');

%% [MODIFICAÇÃO 3]: FASE 1 - PRÉ-TREINAMENTO APENAS COM DADOS (WARM-UP)
% disp('Fase 1: Pré-ajuste baseado apenas em dados (Warm-up)...');
% epochs_warmup = 50; %00;
% for epoch = 1:epochs_warmup
%     % M1 (lambda_phys = 0)
%     [~,~, ~, gradNet_M1, g_ueq1, g_b1] = dlfeval(@lossM1,...
%         net_M1, dlX, dlU, ueq_M1, b_M1, h_norm, 0, lambda_ic, t_max);
%     [net_M1, avg_M1, avgSq_M1] = adamupdate(net_M1, gradNet_M1, avg_M1, avgSq_M1, epoch, 1e-3);
%     ueq_M1 = ueq_M1 - 1e-4 * g_ueq1;
%     b_M1   = b_M1   - 1e-5 * g_b1;
% 
%     % M2 (lambda_phys = 0)
%     [~,~, ~, gradNet_M2, g_ueq2, g_b2, g_alpha2] = dlfeval(@lossM2,...
%         net_M2, dlX, dlU, ueq_M2, b_M2, alpha_M2, h_norm, 0, lambda_ic, t_max);
%     [net_M2, avg_M2, avgSq_M2] = adamupdate(net_M2, gradNet_M2, avg_M2, avgSq_M2, epoch, 1e-3);
%     ueq_M2   = ueq_M2   - 1e-4 * g_ueq2;
%     b_M2     = b_M2     - 1e-5 * g_b2;
%     alpha_M2 = alpha_M2 - 1e-4 * g_alpha2;
% end

%% FASE 2: TREINAMENTO PINN COMPLETO (DADOS + FÍSICA)
disp('Treinamento concorrente dos Modelos M1 e M2 com Física...');
% Reset das estatísticas do Adam para a nova fase
avg_M1 = []; avgSq_M1 = [];
avg_M2 = []; avgSq_M2 = [];

nj=1;
for epoch = 1:numEpochs
    % ---------- TREINAMENTO M1 (EDO - 5 Pontos) ----------
    [loss_M1, loss1_M1, loss2_M1, gradNet_M1, g_ueq1, g_b1] = dlfeval(@lossM1,...
        net_M1, dlX, dlU, ueq_M1, b_M1, h_norm, lambda_phys, lambda_ic, t_max);
    
    [net_M1, avg_M1, avgSq_M1] = adamupdate(net_M1, gradNet_M1, avg_M1, avgSq_M1, epoch, learnRate);
    
    ueq_M1 = ueq_M1 - lr_ueq * g_ueq1;
    b_M1   = b_M1   - lr_b   * g_b1;
  
    % ---------- TREINAMENTO M2 (EDF - Caputo L1) ----------
    [loss_M2, loss1_M2, loss2_M2, gradNet_M2, g_ueq2, g_b2, g_alpha2] = dlfeval(@lossM2,...
        net_M2, dlX, dlU, ueq_M2, b_M2, alpha_M2, h_norm, lambda_phys, lambda_ic, t_max);
    
    [net_M2, avg_M2, avgSq_M2] = adamupdate(net_M2, gradNet_M2, avg_M2, avgSq_M2, epoch, learnRate);
    
    ueq_M2   = ueq_M2   - lr_ueq   * g_ueq2;
    b_M2     = b_M2     - lr_b     * g_b2;
    alpha_M2 = alpha_M2 - lr_alpha * g_alpha2;
    
    % Restrições físicas
    ueq_M1 = min(max(ueq_M1, dlarray(ueq_min)), dlarray(ueq_max));
    b_M1   = min(max(b_M1,   dlarray(b_min)),   dlarray(b_max));
    
    ueq_M2   = min(max(ueq_M2,   dlarray(ueq_min)), dlarray(ueq_max));
    b_M2     = min(max(b_M2,     dlarray(b_min)),   dlarray(b_max));
    alpha_M2 = min(max(alpha_M2, dlarray(alpha_min)), dlarray(alpha_max));
   
    if mod(epoch, 200) == 0 %200
        fprintf('Época %d | ueq: %.3e  | b: %.3e | alpha: %.3f  | L_M2: %.1e (%.1e)\n', ...
            epoch, ueq_M2, b_M2, alpha_M2, loss_M2, lambda_phys*loss2_M2/loss1_M2);
    end
    p11(nj)=ueq_M1;
    p21(nj)=b_M1;
    p12(nj)=ueq_M2;
    p22(nj)=b_M2;
    p32(nj)=alpha_M2;
E1a(nj)=loss_M1; 
E2a(nj)= loss1_M1; 
E3a(nj)=loss2_M1; 
E1b(nj)=loss_M2; 
E2b(nj)=loss1_M2;
E3b(nj)=loss2_M2; 
     nj=nj+1;
end

%% 4. AVALIAÇÃO DOS RESULTADOS E SELEÇÃO DE MODELOS
u_pred_M1 = extractdata(predict(net_M1, dlX))';
u_pred_M2 = extractdata(predict(net_M2, dlX))';

% Parâmetros físicos ajustados
ueq_ajust1   = extractdata(ueq_M1); 
b_ajust1     = extractdata(b_M1);  
ueq_ajust2   = extractdata(ueq_M2); 
b_ajust2     = extractdata(b_M2);  
alpha_ajust2 = extractdata(alpha_M2);
tau_ajust2   = 1;

% Curvas ajustadas geradas a partir dos parâmetros
u_ext1_fit = ueq_ajust1*(1 - exp(-b_ajust1*x_data));
u_ext2_fit = ueq_ajust2*(1 - mlf(alpha_ajust2,1,-b_ajust2*tau_ajust2*(x_data/tau_ajust2).^alpha_ajust2,16));

%% =========================================================
% 5. CRITÉRIOS DE SELEÇÃO DE MODELOS (AIC / BIC)
%% =========================================================
res_M1 = u_exp(:) - u_ext1_fit(:);
res_M2 = u_exp(:) - u_ext2_fit(:);
 
MSE_M1 = mean(res_M1.^2);
MSE_M2 = mean(res_M2.^2);

var_M1 = var(res_M1, 1);
var_M2 = var(res_M2, 1);

p_M1 = 2;  % M1: ueq, b
p_M2 = 3;  % M2: ueq, b, alpha

nBins = 10;  
H_M1 = shannonEntropyResiduals(res_M1, nBins);
H_M2 = shannonEntropyResiduals(res_M2, nBins);

tiny = realmin('double');

AIC_M1 = N * log(max(MSE_M1, tiny)) + 2 * p_M1;
AIC_M2 = N * log(max(MSE_M2, tiny)) + 2 * p_M2;
BIC_M1 = N * log(max(var_M1, tiny)) + p_M1 * log(N);
BIC_M2 = N * log(max(var_M2, tiny)) + p_M2 * log(N);

AICH_M1 = N * log(max(MSE_M1 / H_M1, tiny)) + 2 * p_M1;
AICH_M2 = N * log(max(MSE_M2 / H_M2, tiny)) + 2 * p_M2;
BICH_M1 = N * log(max(var_M1 / H_M1, tiny)) + p_M1 * log(N);
BICH_M2 = N * log(max(var_M2 / H_M2, tiny)) + p_M2 * log(N);

%% Comparação: modelo físico vs. PINN
res_rede_M1 = u_exp(:) - u_pred_M1(:);
res_rede_M2 = u_exp(:) - u_pred_M2(:);
 
MSE_rede_M1 = mean(res_rede_M1.^2);
MSE_rede_M2 = mean(res_rede_M2.^2);
ganho_M1 = 100*(MSE_M1-MSE_rede_M1)/MSE_M1;
ganho_M2 = 100*(MSE_M2-MSE_rede_M2)/MSE_M2;

%% Relatório
%% =========================================================
% RELATÓRIOS E IMPRESSÃO DAS TABELAS
%% =========================================================
fprintf('\n=======================================================\n');
fprintf('\nINFORMAÇÕES\n');
fprintf('\n=======================================================\n');
fprintf('EXP. Dados =   %s \n', EXPDADOS );
fprintf('lambda =   %.1f \n', lambda_phys);
fprintf('Epocas =   %.2f  \n', numEpochs  );
fprintf('neuronios =   %.1f \n', ARQ );
fprintf('learnRate =   %.2e \n', learnRate );
fprintf('\n M1 =====\n');
fprintf('ueq_chute =   %.2e  \n', ueq_chute1);
fprintf('b_chute  =   %.2e  \n', b_chute1);
fprintf(' MSE(chute) =   %.2e \n', MSEx_M1 );
fprintf('ueq  =   %.2e  \n', ueq_ajust1);
fprintf('b  =   %.2e  \n', b_ajust1);
fprintf(' MSE(phys) =   %.2e \n', MSE_M1 );
fprintf(' MSE(pinn) =   %.2e \n', MSE_rede_M1 );
fprintf(' AIC =   %.2f \n', AIC_M1 );
fprintf(' BIC =   %.2f \n', BIC_M1 );
fprintf(' H =     %.2f \n', H_M1 );
fprintf(' AICH =   %.2f \n', AICH_M1 );
fprintf(' BICH =   %.2f \n', BICH_M1 );
fprintf(' ganho =   %.2f %% \n', ganho_M1 );
fprintf(' loss i =   %.2e \n', E1a(1) );
fprintf(' loss i(data) =   %.2e \n', E2a(1) );
fprintf(' loss i(phys) =   %.2e \n', E3a(1) );
fprintf(' loss =   %.2e \n', loss_M1 );
fprintf(' loss(data) =   %.2e \n', loss1_M1 );
fprintf(' loss(phys) =   %.2e \n', loss2_M1 );
fprintf(' peso =   %.2e \n', lambda_phys*loss2_M1/loss1_M1 );
fprintf('\n M2 =====\n');
fprintf('ueq_chute  =   %.2e  \n', ueq_chute2);
fprintf('b_chute  =   %.2e  \n', b_chute2);
fprintf('alpha_chute  =   %.2e  \n', alpha_chute2);
fprintf(' MSE(chute) =   %.2e \n', MSEx_M2 );
fprintf('ueq  =   %.2e  \n', ueq_ajust2);
fprintf('b  =   %.2e  \n', b_ajust2);
fprintf('alpha  =   %.2e  \n', alpha_ajust2);
fprintf(' MSE(phys) =   %.2e \n', MSE_M2 );
fprintf(' MSE(pinn) =   %.2e \n', MSE_rede_M2 );
fprintf(' AIC =   %.2f \n', AIC_M2 );
fprintf(' BIC =   %.2f \n', BIC_M2 );
fprintf(' H =     %.2f \n', H_M2 );
fprintf(' AICH =   %.2f \n', AICH_M2 );
fprintf(' BICH =   %.2f \n', BICH_M2 );
fprintf(' ganho =   %.2f %% \n', ganho_M2 );
fprintf(' loss i =   %.2e \n', E1b(1) );
fprintf(' loss i(data) =   %.2e \n', E2b(1) );
fprintf(' loss i(phys) =   %.2e \n', E3b(1) );
fprintf(' loss =   %.2e \n', loss_M2 );
fprintf(' loss(data) =   %.2e \n', loss1_M2 );
fprintf(' loss(phys) =   %.2e \n', loss2_M2 );
fprintf(' peso =   %.2e \n', lambda_phys*loss2_M2/loss1_M2 );
fprintf('\nComparação =====\n');
fprintf(' DAIC(1-2) =   %.2f \n', AIC_M1-AIC_M2 );
fprintf(' Modelo preferido por AIC: %s\n', ternary(AIC_M1 < AIC_M2, 'M1', 'M2'));
fprintf(' DBIC(1-2) =   %.2f \n', BIC_M1-BIC_M2 );
fprintf(' Modelo preferido por BIC: %s\n', ternary(BIC_M1 < BIC_M2, 'M1', 'M2'));
fprintf(' DAICH(1-2) =   %.2f \n', AICH_M1-AICH_M2 );
fprintf(' Modelo preferido por AICH: %s\n', ternary(AICH_M1 < AICH_M2, 'M1', 'M2'));
fprintf(' DBIC(1-2) =   %.2f \n', BICH_M1-BICH_M2 );
fprintf(' Modelo preferido por BICH: %s\n', ternary(BICH_M1 < BICH_M2, 'M1', 'M2'));
fprintf(' H1/H2 =   %.2e \n', H_M1/H_M2 );
fprintf(' Modelo preferido por MSE(phys): %s\n', ternary(MSE_M1 < MSE_M2, 'M1', 'M2'));
fprintf(' Modelo preferido por MSE(pinn): %s\n', ternary(MSE_rede_M1 < MSE_rede_M2, 'M1', 'M2'));
fprintf('\n=======================================================\n');

%% FIGURA
figure(1);
plot(x_data, u_ext1_fit, 'y-'); hold on;
plot(x_data, u_ext2_fit, 'g-'); 
plot(x_data, u_exp, 'go', 'MarkerFaceColor', [0.8 0.8 0.8]); 
plot(x_data, u_pred_M1, 'b-', 'LineWidth', 2);
plot(x_data, u_pred_M2, 'r--', 'LineWidth', 2);
plot(x_data, u_chute1, 'bs-', 'LineWidth', 1);
plot(x_data, u_chute2, 'rs--', 'LineWidth', 1);
%legend('Exponencial','Fracionário','Dados Experimentais', '\mathcal{M}_1 (EDO Langmuir)', '\mathcal{M}_2 (EDF Caputo)', 'Interpreter', 'tex');
legend('Exponencial','Fracionário','Dados Simulados com Ruído', ...
    '\mathcal{M}_1 (EDO Langmuir - treino)', '\mathcal{M}_2 (EDF Caputo - treino)', ...
    '\mathcal{M}_1 (EDO Langmuir) - chute', '\mathcal{M}_2 (EDF Caputo) - chute', 'Interpreter', 'tex');
xlabel('Tempo (x)'); ylabel('Concentração Adsorvida (u)');
title('Seleção de Modelos via PINNs: EDO Clássica vs. EDF de Caputo');
grid on;

%% =========================================================
% FUNÇÕES DE PERDA (LOSS FUNCTIONS)
%% =========================================================

%% [MODIFICAÇÃO 4]: lossM1 com escala de tempo ajustada
function [loss, loss_data, loss_phys, grad_net, grad_ueq, grad_b] = lossM1(net, dlX, dlU, ueq, b, h_norm, lambda_phys, lambda_ic, t_max)
    u_pred = forward(net, dlX);
    
    % 1. Loss dos Dados
    loss_data = mean((u_pred - dlU).^2);
    
    % 2. Loss da Física (M1: EDO de Langmuir)
    N = size(u_pred, 2);
    du_dx_norm = zeros(1, N, 'like', u_pred);
    
    du_dx_norm(3:N-2) = (-u_pred(5:N) + 8*u_pred(4:N-1) - 8*u_pred(2:N-3) + u_pred(1:N-4)) / (12*h_norm);
    du_dx_norm(1) = (u_pred(2) - u_pred(1))/h_norm;
    du_dx_norm(2) = (u_pred(3) - u_pred(2))/h_norm;
    du_dx_norm(N-1) = (u_pred(N) - u_pred(N-1))/h_norm;
    du_dx_norm(N) = (u_pred(N) - u_pred(N-1))/h_norm;
    
    % Aplicação da regra da cadeia para tempo real: du/dt = (1/t_max) * du/dt_norm
    du_dx = du_dx_norm / t_max;
    
    % Condição inicial u(0) = 0
    u0 = 0;
    loss_ic = mean((u_pred(1) - u0).^2);
    
    % Equação governante M1
    physics_res = du_dx - b*(ueq - u_pred);
    loss_phys = mean(physics_res.^2);
    
    % Loss Total Balanceada
    loss = loss_data + lambda_phys * loss_phys + lambda_ic * loss_ic;
    
    % Gradientes via Autodiff
    [grad_net, grad_ueq, grad_b] = dlgradient(loss, net.Learnables, ueq, b);
end

%% [MODIFICAÇÃO 5]: lossM2 vetorizada com escala de tempo ajustada
function [loss, loss_data, loss_phys, grad_net, grad_ueq, grad_b, grad_alpha] = lossM2(net, dlX, dlU, ueq, b, alpha, h_norm, lambda_phys, lambda_ic, t_max)
    u_pred = forward(net, dlX);
    
    % 1. Loss dos Dados
    loss_data = mean((u_pred - dlU).^2);
    
    % 2. Aproximação diferenciável da Função Gamma
    x_gamma = 2 - alpha;
    t_g = x_gamma - 1;
    c1 = -0.577191652; c2 = 0.988205891; c3 = -0.897056937; c4 = 0.918206857;
    c5 = -0.756704078; c6 = 0.482199394; c7 = -0.193527818; c8 = 0.035868343;
    gamma_approx = 1 + c1*t_g + c2*t_g.^2 + c3*t_g.^3 + c4*t_g.^4 + c5*t_g.^5 + c6*t_g.^6 + c7*t_g.^7 + c8*t_g.^8;
    
    % 3. Esquema L1 vetorizado com tempo adimensionado
    N = size(u_pred, 2);
    diff_u = u_pred(2:end) - u_pred(1:end-1);
    
    r = (1:N-1)';
    c = (1:N-1);
    R = repmat(r, 1, N-1);
    C = repmat(c, N-1, 1);
    
    base1 = R - C + 1;
    base2 = R - C;
    
    invalid = C > R;
    base1(invalid) = 1;
    base2(invalid) = 1;
    
    W = (base1.^(1-alpha) - base2.^(1-alpha)) .* ~invalid;
    
    coef = 1.0 / (gamma_approx * (h_norm^alpha));
    D_alpha_norm = coef * (W * diff_u(:));
    D_alpha_u = [dlarray(0), D_alpha_norm.'];
    
    % Aplicação da regra da cadeia para tempo real: D_t^\alpha u = (1 / t_max^\alpha) * D_\tilde{t}^\alpha u
    D_alpha_phys = D_alpha_u(2:end) / (t_max^alpha);
    
    % Condição inicial u(0) = 0
    u0 = 0;
    loss_ic = mean((u_pred(1) - u0).^2);
    
    % Equação governante M2
    tau = 1;
    physics_res = D_alpha_phys - (b/tau^(alpha-1))*(ueq - u_pred(2:end));
    loss_phys = mean(physics_res.^2);
    
    % Loss Total Balanceada
    loss = loss_data + lambda_phys * loss_phys + lambda_ic * loss_ic;
    
    % Gradientes via Autodiff
    [grad_net, grad_ueq, grad_b, grad_alpha] = dlgradient(loss, net.Learnables, ueq, b, alpha);
end

function H = shannonEntropyResiduals(residuals, nBins)
    residuals = residuals(:);
    escala = max(abs(residuals));
    if escala <= eps
        H = eps;
        return;
    end
    residualsNorm = residuals / escala;
    edges = linspace(-1, 1, nBins + 1);
    counts = histcounts(residualsNorm, edges);
    p = counts(counts > 0) / sum(counts);
    H = -sum(p .* log(p));
    H = max(H, eps);
end

function resultado = ternary(condicao, valorVerdadeiro, valorFalso)
    if condicao
        resultado = valorVerdadeiro;
    else
        resultado = valorFalso;
    end
end