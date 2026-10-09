import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Configurações Globais e Reprodutibilidade
SEED = 33
EPOCHS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car.data" if (BASE_DIR / "car.data").exists() else Path("car.data")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 2. Prevenção de Fuga de Dados e Divisão Estratificada
def preparar_dados():
    with mlflow.start_span(name="preparar"):
        df = pd.read_csv(
            DATA_PATH,
            header=None,
            names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
        )
        y = LabelEncoder().fit_transform(df["class"])
        X_raw = df.drop("class", axis=1)

        # Divisão ANTES do OneHotEncoder
        X_tr_raw, X_tmp_raw, y_tr, y_tmp = train_test_split(
            X_raw, y, test_size=0.4, random_state=SEED, stratify=y
        )
        X_val_raw, X_teste_raw, y_val, y_teste = train_test_split(
            X_tmp_raw, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp
        )

        # Ajuste exclusivo no Treino
        enc = OneHotEncoder(sparse_output=False, dtype=np.float32)
        X_tr = enc.fit_transform(X_tr_raw)
        X_val = enc.transform(X_val_raw)
        X_teste = enc.transform(X_teste_raw)

        # Guardar artefacto de pré-processamento
        resumo = {
            "amostras_treino": len(y_tr),
            "amostras_validacao": len(y_val),
            "amostras_teste": len(y_teste),
        }
        resumo_path = BASE_DIR / "resumo_prep.json"
        with open(resumo_path, "w", encoding="utf-8") as f:
            json.dump(resumo, f, indent=2)

    return X_tr, y_tr, X_val, y_val, X_teste, y_teste


# 3. Definição da Arquitetura MLP
class MLPCar(nn.Module):
    def __init__(self, in_dim, drop_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        return self.net(x)


def main():
    mlflow.set_experiment("A1_MLP_Car_3_Runs")
    X_tr, y_tr, X_val, y_val, X_te, y_te = preparar_dados()

    # Converter para Tensores do PyTorch
    T = lambda x: torch.tensor(x, dtype=torch.float32, device=DEVICE)
    Tl = lambda x: torch.tensor(x, dtype=torch.long, device=DEVICE)
    x_tr, y_tr = T(X_tr), Tl(y_tr)
    x_val, y_val = T(X_val), Tl(y_val)
    x_te, y_te = T(X_te), Tl(y_te)

    # 4. As 3 Runs Focadas: Base, Regularizada e Otimizada
    configs = [
        {
            "nome": "Run 1 - Modelo Base",
            "slug": "run_1_base",
            "lr": 0.01,
            "l2": 0.0,
            "drop": 0.0,
            "ideia": "Comportamento natural da rede sem restrições de L2 ou Dropout.",
        },
        {
            "nome": "Run 2 - Modelo Regularizado",
            "slug": "run_2_regularizado",
            "lr": 0.01,
            "l2": 1e-5,
            "drop": 0.0,
            "ideia": "Controlo de magnitude dos pesos através de regularização L2.",
        },
        {
            "nome": "Run 3 - Modelo Otimizado",
            "slug": "run_3_otimizado",
            "lr": 0.01,
            "l2": 1e-5,
            "drop": 0.2,
            "ideia": "Resiliência contra co-adaptação e atalhos usando L2 e Dropout de 20%.",
        },
    ]

    melhor_f1_global = -1.0
    melhor_estado_global = None
    melhor_nome = ""
    historico_runs = {}

    configs_completas = [
        {
            **cfg,
            "epocas": EPOCHS,
            "semente": SEED,
            "split": "Estratificado 60/20/20",
            "arquitetura": "Linear(16)->ReLU->Drop->Linear(4)",
        }
        for cfg in configs
    ]
    configs_path = BASE_DIR / "configs_runs.json"
    with open(configs_path, "w", encoding="utf-8") as f:
        json.dump(configs_completas, f, indent=2, ensure_ascii=False)

    for cfg in configs:
        with mlflow.start_run(run_name=cfg["nome"]):
            # Tracking de Parâmetros
            mlflow.log_params({
                "taxa": cfg["lr"],
                "L2": cfg["l2"],
                "dropout": cfg["drop"],
                "epocas": EPOCHS,
                "arquitetura": "Linear(16)->ReLU->Drop->Linear(4)",
                "semente": SEED,
                "split": "Estratificado 60/20/20",
                "ideia": cfg["ideia"],
            })

            modelo = MLPCar(X_tr.shape[1], cfg["drop"]).to(DEVICE)
            criterio = nn.CrossEntropyLoss()
            otimizador = optim.Adam(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["l2"])

            perdas_tr, perdas_v = [], []
            melhor_f1_run = -1.0
            melhor_estado_run = None

            with mlflow.start_span(name="treinar_e_validar"):
                for ep in range(EPOCHS):
                    # Fase: Treinar
                    modelo.train()
                    otimizador.zero_grad()
                    p_tr = criterio(modelo(x_tr), y_tr)
                    p_tr.backward()
                    otimizador.step()

                    # Fase: Validar
                    modelo.eval()
                    with torch.no_grad():
                        out_v = modelo(x_val)
                        p_v = criterio(out_v, y_val)
                        prev = out_v.argmax(1).cpu().numpy()
                        acc = accuracy_score(y_val.cpu().numpy(), prev)
                        f1 = f1_score(y_val.cpu().numpy(), prev, average="macro")

                    perdas_tr.append(p_tr.item())
                    perdas_v.append(p_v.item())

                    # Tracking de Métricas por Época
                    mlflow.log_metrics({
                        "train_loss": p_tr.item(),
                        "val_loss": p_v.item(),
                        "val_accuracy": acc,
                        "val_f1_macro": f1,
                    }, step=ep)

                    if f1 > melhor_f1_run:
                        melhor_f1_run = f1
                        melhor_estado_run = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}

            historico_runs[cfg["nome"]] = {"perdas_tr": perdas_tr, "perdas_v": perdas_v}

            # Gerar Gráfico de Linha de Perda individual
            plt.figure(figsize=(7, 4))
            plt.plot(perdas_tr, label="Perda de Treino")
            plt.plot(perdas_v, label="Perda de Validação")
            plt.title(f"Curva de Perda - {cfg['nome']}")
            plt.xlabel("Épocas")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.tight_layout()
            grafico_path = str(BASE_DIR / f"curva_{cfg['slug']}.png")
            plt.savefig(grafico_path)
            plt.close()

            # Guardar os 3 Artefactos exigidos
            caminho_modelo = str(BASE_DIR / f"modelo_{cfg['slug']}.pt")
            torch.save(melhor_estado_run, caminho_modelo)
            mlflow.log_artifact(caminho_modelo)
            mlflow.log_artifact(str(BASE_DIR / "resumo_prep.json"))
            mlflow.log_artifact(str(configs_path))
            mlflow.log_artifact(grafico_path)

            # Verificar se é o vencedor global (para o Teste Final)
            if melhor_f1_run > melhor_f1_global:
                melhor_f1_global = melhor_f1_run
                melhor_estado_global = melhor_estado_run
                melhor_nome = cfg["nome"]

    # Gerar Gráfico Comparativo das 3 Runs
    plt.figure(figsize=(8, 5))
    for nome_run, perdas in historico_runs.items():
        plt.plot(perdas["perdas_v"], label=f"Validação: {nome_run}")
    plt.title("Comparação de Perda de Validação entre as 3 Runs")
    plt.xlabel("Épocas")
    plt.ylabel("Cross-Entropy Loss (Validação)")
    plt.legend()
    plt.tight_layout()
    grafico_comp = str(BASE_DIR / "comparacao_curvas_perda.png")
    plt.savefig(grafico_comp)
    plt.close()

    # 5. Avaliação Única no Teste (mantido isolado até agora)
    print(f"\nModelo Selecionado pela Validação: {melhor_nome} (F1 Val: {melhor_f1_global:.4f})")

    modelo_final = MLPCar(X_tr.shape[1], 0.0).to(DEVICE)  # Dropout desligado na inferência
    modelo_final.load_state_dict(melhor_estado_global)
    modelo_final.eval()

    with torch.no_grad():
        out_te = modelo_final(x_te)
        prev_te = out_te.argmax(1).cpu().numpy()
        y_te_np = y_te.cpu().numpy() if isinstance(y_te, torch.Tensor) else y_te
        acc_teste = accuracy_score(y_te_np, prev_te)
        f1_teste = f1_score(y_te_np, prev_te, average="macro")

        print("\n--- Resultados do Teste Final (Avaliados 1 única vez) ---")
        print(f"Acurácia no Teste: {acc_teste:.4f}")
        print(f"F1 Macro no Teste: {f1_teste:.4f}")


if __name__ == "__main__":
    main()
