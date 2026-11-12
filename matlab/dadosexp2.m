%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parâmetros físicos verdadeiros (para gerar dados)
% M1
ueq_true1 = 0.082; b_true1 = 0.0030; 
% M2
ueq_true2 = 0.091; b_true2 = 0.010; alpha_true2 = 0.78; 

% Geração de dados (comportamento de Langmuir clássico + ruído)
tau_true2=1;
u_ext1 = ueq_true1*(1 - exp(-b_true1*x_data));
u_ext2 = ueq_true2*(1 - mlf(alpha_true2,1,-b_true2*tau_true2*(x_data/tau_true2).^alpha_true2,12));

plot(x_data,u_ext1,x_data,u_ext2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%