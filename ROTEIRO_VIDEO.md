# Roteiro para o Vídeo de Demonstração (3 Runs: Base, Regularizada e Otimizada)

Utilize os resultados reais gerados no seu MLflow (`http://127.0.0.1:5000`) e os artefatos salvos no diretório do projeto.

---

## 1. Apresentação do Problema e Dados (1 a 2 min)
- **Dataset Tabular:** Car Evaluation (`car.data`), 1.728 instâncias, 6 atributos categóricos (`buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`) e 4 classes de aceitabilidade do veículo (`unacc`, `acc`, `good`, `vgood`).
- **Prevenção de Fuga de Dados (Data Leakage):**
  - Os dados foram divididos em treino (60% - 1.036 amostras), validação (20% - 346 amostras) e teste (20% - 346 amostras) com estratificação (`stratify`) para preservar a proporção exata de cada classe.
  - O `OneHotEncoder` foi ajustado (`fit`) **estritamente nos dados de treino**, transformando depois validação e teste com `transform`. O codificador e a rede nunca "espreitaram" dados de validação ou teste.
- **Semente Fixa:** Uso de semente fixa (`seed = 33`) para garantir reprodutibilidade exata em todas as divisões e inicializações.

---

## 2. Desenho Experimental: As 3 Runs Focadas
Apresente no código a lista de configurações e explique a hipótese por trás de cada uma:

1. **Run 1 - Modelo Base (Referência):**
   - `LR: 0.01` | `L2: 0.0` | `Dropout: 0.0`
   - *Ideia:* Estabelecer o comportamento natural da rede. Sem restrições de L2 ou Dropout, a rede aprende rapidamente. O objetivo é observar o seu desempenho puro e identificar potenciais sinais de *overfitting*, servindo como ponto de comparação para as configurações subsequentes.

2. **Run 2 - Modelo Regularizado (L2):**
   - `LR: 0.01` | `L2: 1e-5` | `Dropout: 0.0`
   - *Ideia:* Introduzir controlo sobre os pesos da rede. Mantendo a taxa de aprendizado alta, a penalização L2 impede que a rede atribua importância excessiva a características isoladas do conjunto de dados, testando se a regularização dos pesos melhora a capacidade de generalização face à Run 1.

3. **Run 3 - Modelo Otimizado (L2 + Dropout):**
   - `LR: 0.01` | `L2: 1e-5` | `Dropout: 0.2`
   - *Ideia:* Forçar a resiliência da rede. Adicionando o Dropout, 20% dos neurônios são desativados aleatoriamente durante o treino. Isto evita dependências fortes entre neurônios específicos (atalhos viciados), promovendo aprendizagem de padrões robustos e independentes para a melhor generalização possível.

---

## 3. Demonstração no MLflow UI (`http://127.0.0.1:5000`)
- **Tracking de Parâmetros:** Mostre na tabela do experimento `A1_MLP_Car_3_Runs` os hiperparâmetros registrados: taxa de aprendizado, L2, dropout, épocas, arquitetura, semente, split e a hipótese/ideia.
- **Tracking de Métricas:** Mostre as métricas por época (`train_loss`, `val_loss`, `val_accuracy`, `val_f1_macro`).
- **Tracing (Spans com `mlflow.start_span`):**
  - Abra a aba **Traces** da run no MLflow.
  - Mostre as etapas rastreadas: `preparar` e `treinar_e_validar`.
- **Artefatos da Rodada:**
  - Mostre os 3 artefatos salvos em cada run: `modelo_run_*.pt`, `resumo_prep.json` e `curva_run_*.png`.

---

## 4. Diagnóstico Visual das Curvas de Perda (Loss Curves)
Mostre os gráficos de linha de perda (sem formato de torre ou barras):
- **Curvas de Perda Individuais (`curva_run_1_base.png`, `curva_run_2_regularizado.png`, `curva_run_3_otimizado.png`):**
  - Mostre a curva de treino vs validação ao longo das 300 épocas.
  - Explique visualmente o comportamento e a estabilidade.
- **Comparação entre as 3 Runs (`comparacao_curvas_perda.png`):**
  - Mostre as curvas de perda de validação das 3 runs sobrepostas.
  - Demonstre a evolução lógica desde o modelo sem restrições até o modelo totalmente otimizado.

---

## 5. Seleção e Avaliação Única no Teste Final
- **Critério de Seleção:** Escolha baseada exclusivamente no melhor F1 macro da partição de validação.
- **Avaliação no Teste:** O modelo vencedor é avaliado uma única vez no conjunto de teste isolado (20%), demonstrando a acurácia e F1 Macro finais sem contaminação prévia.
