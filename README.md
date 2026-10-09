# MLP no Car Evaluation com PyTorch + MLflow

Classificação de aceitabilidade veicular no dataset tabular **Car Evaluation** utilizando MLP em PyTorch com rastreamento de experimentos (**tracking**) e fases do pipeline (**tracing**) integrados ao MLflow.

---

## 1. Princípios do Experimento

* **Prevenção de Fuga de Dados (Data Leakage):**
  Os dados foram separados em treino (60%), validação (20%) e teste (20%) antes do ajuste do `OneHotEncoder`, garantindo que o codificador e o modelo nunca acessem as partições de validação ou teste.
* **Divisão Estratificada e Sementes:**
  A divisão dos dados manteve a proporção original das classes (`stratify`) e utilizou semente fixa (`seed = 33`), garantindo reprodutibilidade exata.
* **Rastreio de Etapas (Tracing):**
  O uso de `mlflow.start_span` documenta no MLflow Tracing as fases fundamentais: `"preparar"` e `"treinar_e_validar"`.
* **Registro de Evidências (Tracking):**
  - **Parâmetros:** Taxa de aprendizado, regularização L2, dropout, épocas, arquitetura, semente e split.
  - **Métricas por época:** `train_loss`, `val_loss`, `val_accuracy` e `val_f1_macro`.
* **Avaliação Única no Teste:**
  O conjunto de teste de 20% foi mantido rigorosamente isolado durante as rodadas e avaliado uma única vez no final, apenas para o modelo vencedor escolhido pela validação.
* **Formato dos Gráficos:**
  Curvas de perda em formato de linha (`curva_*.png` e `comparacao_curvas_perda.png`), sem gráficos de barras/torres.

---

## 2. Desenho Experimental: As 3 Runs Focadas

| Run | Nome | Taxa (LR) | Reg. L2 | Dropout | Ideia / Hipótese |
|---|---|:---:|:---:|:---:|---|
| **1** | `Run 1 - Modelo Base` | `0.01` | `0.0` | `0.0` | **Referência / Baseline:** Estabelecer o comportamento natural da rede sem restrições. A rede aprende rápido e permite observar o desempenho puro e identificar riscos de sobreajuste (*overfitting*). |
| **2** | `Run 2 - Modelo Regularizado` | `0.01` | `1e-5` | `0.0` | **Controlo de Pesos:** Mantém a taxa de aprendizado alta e adiciona penalização L2 suave para evitar que a rede dê peso desproporcional a atributos isolados. |
| **3** | `Run 3 - Modelo Otimizado` | `0.01` | `1e-5` | `0.2` | **Resiliência e Robustez:** L2 somado a 20% de Dropout desativando neurónios aleatoriamente a cada época para eliminar atalhos viciados e maximizar a generalização. |

---

## 3. Instalação e Execução

### Pré-requisitos
No PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Iniciar o Servidor MLflow (Terminal 1)
```powershell
mlflow server --host 127.0.0.1 --port 5000
```
Acesse `http://127.0.0.1:5000` no seu navegador.

### Executar o Experimento (Terminal 2)
```powershell
python mlp_car.py
```

---

## 4. Estrutura de Artefatos Gerados

A cada execução, o projeto e o MLflow registram:
* `modelo_run_*.pt`: pesos do modelo com melhor F1 macro na validação para cada run.
* `resumo_prep.json`: resumo das amostras de treino (1.036), validação (346) e teste (346).
* `curva_run_*.png`: gráficos de linha com `train_loss` vs `val_loss` por época para cada run.
* `comparacao_curvas_perda.png`: gráfico de linha comparativo das perdas de validação entre as 3 runs.
* Registro no MLflow da avaliação única e definitiva no conjunto de teste para o modelo vencedor.
