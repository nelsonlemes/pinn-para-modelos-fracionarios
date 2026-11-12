clear; clc; close all;
rng(42,'twister');

%% 1. GERAÇÃO DOS DADOS SIMULADOS
N = 80; %30;    
TLIMITE = 680; %260; 
MODELO='gM2';
x_data = linspace(0.01, TLIMITE, N)'; 
h = x_data(2) - x_data(1); 

% Parâmetros físicos verdadeiros (para geração de dados)
% M2
ueq_true2 = 0.086; b_true2 = 0.012; alpha_true2 = .95; 
tau_true2 = 1;

% Soluções analíticas exatas para geração dos dados
%u_gera = ueq_true2*(1 - exp(-b_true2*x_data));
%u_gera_teste = ueq_true2*(1 - exp(-b_true2*x_data_teste));
u_gera1 = ueq_true2*(1 - mlf(1,1,-b_true2*tau_true2*(x_data/tau_true2),16));
u_gera = ueq_true2*(1 - mlf(alpha_true2,1,-b_true2*tau_true2*(x_data/tau_true2).^alpha_true2,16));

% Configuração de ruído e seleção do modelo gerador
epsilon =0.001/66.4844; 

u_exp = u_gera1 + epsilon*randn(N,1);

erroe = mean(abs((u_exp - u_gera1)./u_gera1))*100;
errox = mean(abs((u_gera1 - u_gera)./u_gera1))*100;

erroe, errox, 'Tecle (->)'
