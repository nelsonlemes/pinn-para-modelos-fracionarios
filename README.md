# PINN PARA MODELOS FRACIONÁRIOS

Este repositório reúne implementações em MATLAB de *Physics-Informed Neural Networks* (PINNs) para comparar modelos de adsorção clássicos e fracionários. O objetivo é ajustar dados de concentração/adsorção, selecionar o modelo mais adequado e estimar parâmetros cinéticos, contrastando a dinâmica de Langmuir de ordem inteira com uma equação diferencial fracionária de Caputo cuja solução envolve a função de Mittag--Leffler.

## Organização

### `matlab/`

- `pinn_exp_2x.m` — PINN aplicada a dados experimentais; treina e compara a rede do modelo clássico (M1) e a do modelo fracionário (M2).
- `pinn8novo_2x.m` — experimento com dados simulados para a comparação entre os dois modelos.
- `sensalpha_2x.m` — análise de sensibilidade à ordem fracionária `alpha`.
- `ajustesimples.m` — ajuste não linear do modelo de Langmuir clássico.
- `ajustesimples2.m` — ajuste não linear baseado na função de Mittag--Leffler.
- `dadosexp.m` e `dadosexp2.m` — reprodução e geração de curvas de referência experimentais/simuladas.
- `graf.m`, `graf_2x.m` e `graf_2xb.m` — scripts de comparação de modelos, visualização de perdas e evolução dos parâmetros de treinamento.
- `mlf.m` — implementação numérica da função de Mittag--Leffler.

### `data/`

- `dados C1-70.csv`, `dados C1-120.csv` e `dados C2-70.csv` — dados tabulares de entrada.
- `dadosC1-70.txt`, `dadosC1-120.txt` e `dadosC2-70.txt` — séries usadas pelos scripts MATLAB.
- `dados_final.txt` e `Dados_final_simulados.txt` — conjuntos consolidados de dados experimentais e simulados.

### `results/`

- `historico_perdas_pinns.txt` e `historico_perdas_pinns2.txt` — históricos de treinamento das PINNs para análise de convergência e visualização das funções de perda.

### `docs/`

- `dadospinn.docx` — documento técnico com parâmetros, resultados de ajuste, métricas de erro e critérios de seleção dos modelos.

## Modelos comparados

- **M1 — modelo clássico:** cinética de Langmuir descrita por uma EDO de primeira ordem.
- **M2 — modelo fracionário:** dinâmica com derivada de Caputo e resposta descrita pela função de Mittag--Leffler.

As PINNs combinam uma perda orientada pelos dados com uma perda física associada à equação governante. Os scripts também reportam métricas como MSE, AIC, BIC e critérios correlatos para apoiar a escolha do modelo.

## Requisitos

- MATLAB com Deep Learning Toolbox e Optimization Toolbox.
- A função `mlf.m` deve estar no caminho do MATLAB. Ela chama `m2v`; caso essa função auxiliar não esteja disponível no ambiente, ela deverá ser adicionada ao caminho antes da execução dos modelos fracionários.

## Como começar

1. Abra o diretório `matlab/` no MATLAB.
2. Adicione `matlab/` e `data/` ao caminho.
3. Execute `pinn_exp_2x.m` para reproduzir o ajuste aos dados experimentais ou `pinn8novo_2x.m` para os experimentos simulados.
4. Use os scripts `graf_2x.m` e `graf_2xb.m` para analisar perdas e parâmetros estimados.
