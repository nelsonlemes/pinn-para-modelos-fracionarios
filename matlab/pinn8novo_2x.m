%% =========================================================
% PINN PARA SELEÇÃO DE MODELOS: EDO Clássica vs EDF Caputo
% LAB QUIMICA-MATEMATICA - NELSON - 2026 (DADOS SIMULADOS)
%% =========================================================
clear; clc; close all;
rng(42,'twister');

%% 1. GERAÇÃO DOS DADOS SIMULADOS
N = 120; %120%; %180;     
TLIMITE =1000; % 1000;%1500;  
MODELO='gM1';
x_data = linspace(0.01, TLIMITE, N)'; 
h = x_data(2) - x_data(1); 

x_data_teste = 0.01 + (TLIMITE - 0.01) * rand(N, 1);
x_data_teste = sort(x_data_teste);
%x_data_teste = x_data;

% Parâmetros físicos verdadeiros (para geração de dados)
% M1/M2
ueq_true2 = 0.09; b_true2 = 0.005; alpha_true2 = 0.95; 
tau_true2 = 1;

% Soluções analíticas exatas para geração dos dados
u_gera = ueq_true2*(1 - exp(-b_true2*x_data));
u_gera_teste = ueq_true2*(1 - exp(-b_true2*x_data_teste));
%u_gera = ueq_true2*(1 - mlf(alpha_true2,1,-b_true2*tau_true2*(x_data/tau_true2).^alpha_true2,16));
%u_gera_teste = ueq_true2*(1 - mlf(alpha_true2,1,-b_true2*tau_true2*(x_data_teste/tau_true2).^alpha_true2,16));

% Configuração de ruído e seleção do modelo gerador
epsilon =(1/9.5506)*0.0001; % %6.2682

u_exp = u_gera + epsilon*randn(N,1);
u_exp_teste = u_gera_teste + epsilon*randn(N,1);
erroe = mean(abs((u_exp - u_gera)./u_gera))*100;
plot(x_data,u_exp)
erroe, 'Tecle (->)'
u_exp(end),sum(u_exp>0.088)
pause

%% ============================================================
% Modelo
% =============================================================

model = @(p,x) p(1)*(1 - exp(-p(2)*x));
p0 = [ueq_true2, b_true2];

lb = [ueq_true2, b_true2]/1000;
ub = [ueq_true2, b_true2]*100;

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
model = @(p,x) p(1)*(1 - mlf(p(3), 1, -p(2)*tau*(x/tau).^p(3), 12));

%% Estimativas iniciais
p0 = [ueq_true2, b_true2, alpha_true2];

%% Limites: ka, kd, umax > 0; 0 < alpha <= 1
% Altere o limite superior de alpha caso seu modelo admita alpha > 1.
lb = [ueq_true2/1000, b_true2/1000, 0.55];
ub = [ueq_true2*100, b_true2*100, 0.999];

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ADIMENSIONAMENTO DO TEMPO
t_max = max(x_data);
x_data_norm = x_data / t_max; % Tempo normalizado em [0, 1]
h_norm = h / t_max;           % Passo de tempo adimensionado

%% 2. ARQUITETURA DA REDE NEURAL E PARÂMETROS
ARQ = 10; %10
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
ueq_min = ueq_true2/1000;   ueq_max = ueq_true2*100;
b_min   = b_true2/1000;  b_max   = b_true2*100;
alpha_min = 0.55;  alpha_max = 0.999;

% Valores iniciais (dlarrays para rastreamento de gradiente)
ueq_M1 = dlarray(ueq_chute1); b_M1 = dlarray(b_chute1); 
ueq_M2 = dlarray(ueq_chute2);  b_M2 = dlarray(b_chute2); alpha_M2 = dlarray(alpha_chute2); 

% Parâmetros de Treinamento
numEpochs = 50000; %50000;
learnRate = 1e-3;
lr_ueq    = 1e-2;
lr_b      = 5e-3;
lr_alpha  = 1e-2;
%learnRate = 1e-2; %-4;
%lr_ueq    = 1e-2; %-2;
%lr_b      = 1e-4; %-4;
%lr_alpha  = 1e1;  %1;

% Pesos de Perda Redimensionados
lambda_phys = 500; %100  % Peso da física
lambda_ic   = 1000; % Peso da condição inicial u(0) = 0

% Otimizador Adam (M1 e M2)
avg_M1 = []; avgSq_M1 = [];
avg_M2 = []; avgSq_M2 = [];

%% 3. LOOP DE TREINAMENTO VARIACIONAL
dlX = dlarray(x_data_norm', 'CB'); % Entradas com tempo normalizado
dlU = dlarray(u_exp', 'CB');

% FASE 2: TREINAMENTO PINN COMPLETO (DADOS + FÍSICA)
disp('Treinamento concorrente dos Modelos M1 e M2 com Física...');
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
   
    if mod(epoch, 200) == 0
        fprintf('Época %d | ueq: %.3e  | b: %.3e | alpha: %.3f  | L_M2: %.1e (%.1e)\n', ...
            epoch, ueq_M2, b_M2, alpha_M2, loss_M2, lambda_phys*loss2_M2/loss1_M2);
    end
    
%%%%%%%%%
% u_pred_M1 = extractdata(predict(net_M1, dlX))';
% u_pred_M2 = extractdata(predict(net_M2, dlX))';
% 
% % Parâmetros físicos ajustados
% ueq_ajust1   = extractdata(ueq_M1); 
% b_ajust1     = extractdata(b_M1);  
% ueq_ajust2   = extractdata(ueq_M2); 
% b_ajust2     = extractdata(b_M2);  
% alpha_ajust2 = extractdata(alpha_M2);
% tau_ajust2   = 1;
% 
% % Curvas ajustadas geradas a partir dos parâmetros
% u_ext1_fitx = ueq_ajust1*(1 - exp(-b_ajust1*x_data));
% u_ext2_fitx = ueq_ajust2*(1 - mlf(alpha_ajust2,1,-b_ajust2*tau_ajust2*(x_data/tau_ajust2).^alpha_ajust2,12));
% 
% u_ext1_fity = ueq_ajust1*(1 - exp(-b_ajust1*x_data_teste));
% u_ext2_fity = ueq_ajust2*(1 - mlf(alpha_ajust2,1,-b_ajust2*tau_ajust2*(x_data_teste/tau_ajust2).^alpha_ajust2,12));
% %%%%%
% res_M1x = u_exp(:) - u_ext1_fitx(:);
% res_M2x = u_exp(:) - u_ext2_fitx(:);
% res_M1y = u_exp_teste(:) - u_ext1_fity(:);
% res_M2y = u_exp_teste(:) - u_ext2_fity(:); 
% 
% MSE_M1x(nj) = mean(res_M1x.^2);
% MSE_M2x(nj) = mean(res_M2x.^2);
% MSE_M1y(nj) = mean(res_M1y.^2);
% MSE_M2y(nj) = mean(res_M2y.^2);
% 
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
u_ext1_fit = ueq_ajust1*(1 - exp(-b_ajust1*x_data_teste));
u_ext2_fit = ueq_ajust2*(1 - mlf(alpha_ajust2,1,-b_ajust2*tau_ajust2*(x_data_teste/tau_ajust2).^alpha_ajust2,16));

u_ext1_fitq = ueq_ajust1*(1 - exp(-b_ajust1*x_data));
u_ext2_fitq = ueq_ajust2*(1 - mlf(alpha_ajust2,1,-b_ajust2*tau_ajust2*(x_data/tau_ajust2).^alpha_ajust2,16));

% Converte para dlarray no formato 'CB' (Channel, Batch)
dlX_teste = dlarray((x_data_teste/t_max)', 'CB');
% Passa pela rede treinada
dlU_M1_teste = predict(net_M1, dlX_teste); % Use predict em vez de forward no final
dlU_M2_teste = predict(net_M2, dlX_teste); % Use predict em vez de forward no final
% Extrai os dados numéricos normais do dlarray
u_pred_M1_teste = extractdata(dlU_M1_teste)';
u_pred_M2_teste = extractdata(dlU_M2_teste)';

%% =========================================================
% 5. CRITÉRIOS DE SELEÇÃO DE MODELOS (AIC / BIC)
%% =========================================================
res_M1 = u_exp_teste(:) - u_ext1_fit(:);
res_M2 = u_exp_teste(:) - u_ext2_fit(:);
res_M1q = u_exp_teste(:) - u_ext1_fitq(:);
res_M2q = u_exp_teste(:) - u_ext2_fitq(:);
 
MSE_M1 = mean(res_M1.^2);
MSE_M2 = mean(res_M2.^2);
MSE_M1q = mean(res_M1.^2);
MSE_M2q = mean(res_M2.^2);

var_M1 = var(res_M1, 1);
var_M2 = var(res_M2, 1);
var_M1q = var(res_M1q, 1);
var_M2q = var(res_M2q, 1);

p_M1 = 2;  % M1: ueq, b
p_M2 = 3;  % M2: ueq, b, alpha

nBins = 10;  
H_M1 = shannonEntropyResiduals(res_M1, nBins);
H_M2 = shannonEntropyResiduals(res_M2, nBins);
H_M1q = shannonEntropyResiduals(res_M1q, nBins);
H_M2q = shannonEntropyResiduals(res_M2q, nBins);

tiny = realmin('double');
AIC_M1 = N * log(max(MSE_M1, tiny)) + 2 * p_M1;
AIC_M2 = N * log(max(MSE_M2, tiny)) + 2 * p_M2;
BIC_M1 = N * log(max(var_M1, tiny)) + p_M1 * log(N);
BIC_M2 = N * log(max(var_M2, tiny)) + p_M2 * log(N);

AIC_M1q = N * log(max(MSE_M1q, tiny)) + 2 * p_M1;
AIC_M2q = N * log(max(MSE_M2q, tiny)) + 2 * p_M2;
BIC_M1q = N * log(max(var_M1q, tiny)) + p_M1 * log(N);
BIC_M2q = N * log(max(var_M2q, tiny)) + p_M2 * log(N);

AICH_M1 = N * log(max(MSE_M1 / H_M1, tiny)) + 2 * p_M1;
AICH_M2 = N * log(max(MSE_M2 / H_M2, tiny)) + 2 * p_M2;
BICH_M1 = N * log(max(var_M1 / H_M1, tiny)) + p_M1 * log(N);
BICH_M2 = N * log(max(var_M2 / H_M2, tiny)) + p_M2 * log(N);

AICH_M1q = N * log(max(MSE_M1q / H_M1, tiny)) + 2 * p_M1;
AICH_M2q = N * log(max(MSE_M2q / H_M2, tiny)) + 2 * p_M2;
BICH_M1q = N * log(max(var_M1q / H_M1, tiny)) + p_M1 * log(N);
BICH_M2q = N * log(max(var_M2q / H_M2, tiny)) + p_M2 * log(N);

%% Comparação: modelo físico vs. PINN
res_rede_M1 = u_exp_teste(:) - u_pred_M1_teste(:);
res_rede_M2 = u_exp_teste(:) - u_pred_M2_teste(:);
 
MSE_rede_M1 = mean(res_rede_M1.^2);
MSE_rede_M2 = mean(res_rede_M2.^2);

ganho_M1 = 100*(MSE_M1 - MSE_rede_M1)/MSE_M1;
ganho_M2 = 100*(MSE_M2 - MSE_rede_M2)/MSE_M2;

resx_M1 = u_exp(:) - u_chute1(:);
resx_M2 = u_exp(:) - u_chute2(:);
 
MSEx_M1 = mean(resx_M1.^2);
MSEx_M2 = mean(resx_M2.^2);

%% =========================================================
% RELATÓRIOS E IMPRESSÃO DAS TABELAS
%% =========================================================
fprintf('\n=======================================================\n');
fprintf('\nINFORMAÇÕES\n');
fprintf('\n=======================================================\n');
fprintf('ERRO =  | %.2f %% \n', erroe );
fprintf('N =  | %.1f \n', N );
fprintf('TLIMITE =  | %.1f \n', TLIMITE);
fprintf('gMODELO =  | %s \n', MODELO);
fprintf('lambda =  | %.1f \n', lambda_phys);
fprintf('Epocas =  | %.2f  \n', numEpochs  );
fprintf('neuronios =  | %.1f \n', ARQ );
fprintf('learnRate =  | %.2e \n', learnRate );
fprintf('\nM1 =====\n');
fprintf('ueq_chute (erro %%) =  | %.2e (%.2f %%) \n', ueq_chute1, 100*abs(ueq_true2-ueq_chute1)/ueq_true2);
fprintf('b_chute (erro %%) =  | %.2e (%.2f %%) \n', b_chute1, 100*abs(b_true2-b_chute1)/b_true2);
fprintf(' MSE(chute) =  | %.2e \n', MSEx_M1 );
fprintf('ueq (erro %%) =  | %.2e (%.2f %%) \n', ueq_ajust1, 100*abs(ueq_true2-ueq_ajust1)/ueq_true2);
fprintf('b (erro %%) =  | %.2e (%.2f %%) \n', b_ajust1, 100*abs(b_true2-b_ajust1)/b_true2);
fprintf(' MSE(phys) =  | %.2e \n', MSE_M1 );
fprintf(' MSE(pinn) =  | %.2e \n', MSE_rede_M1 );
fprintf(' AIC =  | %.2f \n', AIC_M1 );
fprintf(' BIC =  | %.2f \n', BIC_M1 );
fprintf(' H =    | %.2f \n', H_M1 );
fprintf(' AICH =  | %.2f \n', AICH_M1 );
fprintf(' BICH =  | %.2f \n', BICH_M1 );
fprintf(' AICq =  | %.2f \n', AIC_M1q );
fprintf(' BICq =  | %.2f \n', BIC_M1q );
fprintf(' Hq =    | %.2f \n', H_M1q );
fprintf(' AICHq =  | %.2f \n', AICH_M1q );
fprintf(' BICHq =  | %.2f \n', BICH_M1q );
fprintf(' ganho =   %.2f %% \n', ganho_M1 );
fprintf(' loss i =   %.2e \n', E1a(1) );
fprintf(' loss i(data) =   %.2e \n', E2a(1) );
fprintf(' loss i(phys) =   %.2e \n', E3a(1) );
fprintf(' loss =   %.2e \n', loss_M1 );
fprintf(' loss(data) =   %.2e \n', loss1_M1 );
fprintf(' loss(phys) =   %.2e \n', loss2_M1 );
fprintf(' peso =  | %.2e \n', lambda_phys*loss2_M1/loss1_M1 );
%fprintf(' razão =  | %.2f  \n', lambda_phys*loss2_M1/loss1_M1);
fprintf('\nM2 =====\n');
fprintf('ueq_chute (erro %%) =  | %.2e (%.2f %%) \n', ueq_chute2, 100*abs(ueq_true2-ueq_chute2)/ueq_true2);
fprintf('b_chute (erro %%) =  | %.2e (%.2f %%) \n', b_chute2, 100*abs(b_true2-b_chute2)/b_true2);
fprintf('alpha_chute (erro %%) =  | %.2e (%.2f %%) \n', alpha_chute2, 100*abs(alpha_true2-alpha_chute2)/alpha_true2);
fprintf(' MSE(chute) =  | %.2e \n', MSEx_M2 );
fprintf('ueq (erro %%) =  | %.2e (%.2f %%) \n', ueq_ajust2, 100*abs(ueq_true2-ueq_ajust2)/ueq_true2);
fprintf('b (erro %%) =  | %.2e (%.2f %%) \n', b_ajust2, 100*abs(b_true2-b_ajust2)/b_true2);
fprintf('alpha (erro %%) =  | %.2e (%.2f %%) \n', alpha_ajust2, 100*abs(alpha_true2-alpha_ajust2)/alpha_true2);
fprintf(' MSE(phys) =  | %.2e \n', MSE_M2 );
fprintf(' MSE(pinn) =  | %.2e \n', MSE_rede_M2 );
fprintf(' AIC =  | %.2f \n', AIC_M2 );
fprintf(' BIC =  | %.2f \n', BIC_M2 );
fprintf(' H =    | %.2f \n', H_M2 );
fprintf(' AICH =  | %.2f \n', AICH_M2q );
fprintf(' BICH =  | %.2f \n', BICH_M2q );
fprintf(' AICq =  | %.2f \n', AIC_M2q );
fprintf(' BICq =  | %.2f \n', BIC_M2q );
fprintf(' Hq =    | %.2f \n', H_M2q );
fprintf(' AICHq =  | %.2f \n', AICH_M2q );
fprintf(' BICHq =  | %.2f \n', BICH_M2q );
fprintf(' ganho =   %.2f %% \n', ganho_M2 );
fprintf(' loss i =   %.2e \n', E1b(1) );
fprintf(' loss i(data) =   %.2e \n', E2b(1) );
fprintf(' loss i(phys) =   %.2e \n', E3b(1) );
fprintf(' loss =   %.2e \n', loss_M2 );
fprintf(' loss(data) =   %.2e \n', loss1_M2 );
fprintf(' loss(phys) =   %.2e \n', loss2_M2 );
fprintf(' peso =  | %.2e \n', lambda_phys*loss2_M2/loss1_M2 );
%fprintf(' razão =  | %.2f  \n', lambda_phys*loss2_M2/loss1_M2);
fprintf('\nComparação =====\n');
fprintf(' DAIC(1-2) =  | %.2f \n', AIC_M1-AIC_M2 );
fprintf(' Modelo preferido por AIC: %s\n', ternary(AIC_M1 < AIC_M2, 'M1', 'M2'));
fprintf(' DBIC(1-2) =  | %.2f \n', BIC_M1-BIC_M2 );
fprintf(' Modelo preferido por BIC: %s\n', ternary(BIC_M1 < BIC_M2, 'M1', 'M2'));
fprintf(' DAICH(1-2) =  | %.2f \n', AICH_M1-AICH_M2 );
fprintf(' Modelo preferido por AICH: %s\n', ternary(AICH_M1 < AICH_M2, 'M1', 'M2'));
fprintf(' DBIC(1-2) =  | %.2f \n', BICH_M1-BICH_M2 );
fprintf(' Modelo preferido por BICH: %s\n', ternary(BICH_M1 < BICH_M2, 'M1', 'M2'));
fprintf(' H1/H2 =  | %.2e \n', H_M1/H_M2 );
fprintf(' Modelo preferido por MSE(phys): %s\n', ternary(MSE_M1 < MSE_M2, 'M1', 'M2'));
fprintf(' Modelo preferido por MSE(pinn): %s\n', ternary(MSE_rede_M1 < MSE_rede_M2, 'M1', 'M2'));
fprintf(' DAIC(1-2) =  | %.2f \n', AIC_M1q-AIC_M2q );
fprintf(' Modelo preferido por AIC: %s\n', ternary(AIC_M1q < AIC_M2q, 'M1', 'M2'));
fprintf(' DBIC(1-2) =  | %.2f \n', BIC_M1q-BIC_M2q );
fprintf(' Modelo preferido por BIC: %s\n', ternary(BIC_M1q < BIC_M2q, 'M1', 'M2'));
fprintf(' DAICH(1-2) =  | %.2f \n', AICH_M1q-AICH_M2q );
fprintf(' Modelo preferido por AICH: %s\n', ternary(AICH_M1q < AICH_M2q, 'M1', 'M2'));
fprintf(' DBIC(1-2) =  | %.2f \n', BICH_M1q-BICH_M2q );
fprintf(' Modelo preferido por BICH: %s\n', ternary(BICH_M1q < BICH_M2q, 'M1', 'M2'));
fprintf(' H1/H2 =  | %.2e \n', H_M1q/H_M2q );
fprintf(' Modelo preferido por MSE(phys): %s\n', ternary(MSE_M1 < MSE_M2, 'M1', 'M2'));
fprintf(' Modelo preferido por MSE(pinn): %s\n', ternary(MSE_rede_M1 < MSE_rede_M2, 'M1', 'M2'));
fprintf('\n=======================================================\n');

%% FIGURA
figure(1);
plot(x_data_teste, u_ext1_fit, 'y-'); hold on;
plot(x_data_teste, u_ext2_fit, 'g-'); 
plot(x_data, u_exp, 'go', 'MarkerFaceColor', [0.8 0.8 0.8]); 
plot(x_data, u_pred_M1, 'b-', 'LineWidth', 2);
plot(x_data, u_pred_M2, 'r--', 'LineWidth', 2);
plot(x_data_teste, u_pred_M1_teste, 'b-', 'LineWidth', 3);
plot(x_data_teste, u_pred_M2_teste, 'r--', 'LineWidth', 3);
legend('Exponencial','Fracionário','Dados Simulados com Ruído', ...
    '\mathcal{M}_1 (EDO Langmuir - treino)', '\mathcal{M}_2 (EDF Caputo - treino)', ...
    '\mathcal{M}_1 (EDO Langmuir) - teste', '\mathcal{M}_2 (EDF Caputo) - teste', 'Interpreter', 'tex');
xlabel('Tempo (x)'); ylabel('Concentração Adsorvida (u)');
title('Seleção de Modelos via PINNs (Dados Simulados)');
grid on;

%% =========================================================
% FUNÇÕES DE PERDA (LOSS FUNCTIONS)
%% =========================================================
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
    
    % Regra da cadeia para tempo real: du/dt = (1/t_max) * du/dt_norm
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
    
    % Regra da cadeia para tempo real: D_t^\alpha u = (1 / t_max^\alpha) * D_\tilde{t}^\alpha u
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