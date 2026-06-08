"""Ataque de inferência de pertencimento (membership inference) — LiRA.

Ataca o modelo realista do projeto (regressão logística, 6 features) SEM alterá-lo.
Usa o ataque calibrado LiRA (Likelihood Ratio Attack, Carlini et al., 2022): em vez
da confiança crua (que confunde exemplo fácil com membro), treina vários modelos de
referência que nunca viram o exemplo para estimar a perda "normal" dele quando NÃO é
membro; se o modelo alvo dá uma perda anormalmente baixa, o exemplo provavelmente
estava no treino.

Alvo = modelo de UM cliente com dados limitados (cenário mais realista para a DP
local: cada cliente protege a própria contribuição, e um cliente pequeno é o mais
suscetível). A DP aplicada é a mesma do projeto (clipping + ruído gaussiano).

Pode rodar sozinho (`python membership_attack.py`) ou ser importado por
`run_experiments.py`.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "client"))
from dp_utils import aplicar_dp, calcular_sigma  # noqa: E402

warnings.filterwarnings("ignore")

DADOS = os.path.join(AQUI, "..", "client", "ai4i2020.csv")
SAIDA = os.path.join(AQUI, "..", "results")

FEATURES = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]

N_CLIENTE = 100      # cliente com dados limitados (cenário realista de FL)
K_REF = 64           # modelos de referência do LiRA
N_SPLITS = 5         # repete a separação para estabilizar a métrica
EPSILONS = [0.5, 1, 2, 4, 8, 16, 32, 64]
CLIP_NORM = 1.0
DELTA = 1e-5
REPET_RUIDO = 10     # média sobre sementes de ruído (DP é aleatório)


def carregar_dados():
    df = pd.read_csv(DADOS)
    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    X = df[FEATURES].values.astype(float)
    return StandardScaler().fit_transform(X), df["Machine failure"].values


def _perda(coef, intercept, X, y):
    # Perda log por amostra do modelo logístico, dado coef_/intercept_
    z = X @ np.asarray(coef).ravel() + float(np.asarray(intercept).ravel()[0])
    p1 = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-12, 1 - 1e-12)
    p_certa = np.where(y == 1, p1, 1 - p1)
    return -np.log(p_certa)


def preparar_alvo():
    # Treina o(s) modelo(s) alvo e as referências do LiRA; devolve uma lista de
    # contextos (um por separação) com a perda "out" esperada de cada exemplo.
    X, y = carregar_dados()
    idx = np.arange(len(X))
    contextos = []
    for s in range(N_SPLITS):
        mem, resto = train_test_split(idx, train_size=N_CLIENTE, stratify=y, random_state=s)
        nao, sombra = train_test_split(resto, train_size=N_CLIENTE, stratify=y[resto], random_state=s)

        alvo = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X[mem], y[mem])
        ev = np.concatenate([mem, nao])
        rotulos = np.concatenate([np.ones(N_CLIENTE), np.zeros(N_CLIENTE)])  # 1 = membro

        # Referências: treinadas em subconjuntos da sombra (nunca veem 'ev')
        out = np.zeros((len(ev), K_REF))
        for k in range(K_REF):
            sub, _ = train_test_split(sombra, train_size=N_CLIENTE, stratify=y[sombra], random_state=1000 * s + k)
            ref = LogisticRegression(class_weight="balanced", max_iter=500).fit(X[sub], y[sub])
            out[:, k] = _perda(ref.coef_, ref.intercept_, X[ev], y[ev])

        contextos.append({
            "X": X[ev], "y": y[ev], "rotulos": rotulos,
            "out_media": out.mean(1), "out_desvio": out.std(1) + 1e-6,
            "coef": alvo.coef_, "intercept": alvo.intercept_,
        })
    return contextos


def _score_lira(ctx, coef, intercept):
    # Quanto menor a perda do alvo frente à distribuição "out", mais provável membro
    perda = _perda(coef, intercept, ctx["X"], ctx["y"])
    return (ctx["out_media"] - perda) / ctx["out_desvio"]


def _avaliar(rotulos, score):
    auc = roc_auc_score(rotulos, score)
    fpr, tpr, _ = roc_curve(rotulos, score)
    acc = float(np.max((tpr + (1 - fpr)) / 2))
    return float(auc), acc


def atacar_sem_dp(contextos):
    aucs, accs = [], []
    for ctx in contextos:
        a, ac = _avaliar(ctx["rotulos"], _score_lira(ctx, ctx["coef"], ctx["intercept"]))
        aucs.append(a); accs.append(ac)
    return float(np.mean(aucs)), float(np.mean(accs))


def atacar_com_dp(contextos, clip, epsilon, delta, repeticoes=REPET_RUIDO):
    # Aplica a DP do projeto aos parâmetros do alvo e reavalia o LiRA (média do ruído)
    aucs, accs = [], []
    for ctx in contextos:
        for r in range(repeticoes):
            coef, intercept, _ = aplicar_dp(ctx["coef"], ctx["intercept"], clip, epsilon, delta,
                                            np.random.default_rng(100 + r))
            a, ac = _avaliar(ctx["rotulos"], _score_lira(ctx, coef, intercept))
            aucs.append(a); accs.append(ac)
    return float(np.mean(aucs)), float(np.mean(accs))


def _estilo_quadrado(ax):
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)


def main():
    os.makedirs(SAIDA, exist_ok=True)
    contextos = preparar_alvo()
    auc0, acc0 = atacar_sem_dp(contextos)
    print(f"[sem DP]  AUC={auc0:.4f}  acc={acc0:.4f}  (LiRA, K={K_REF} refs, cliente={N_CLIENTE})")

    linhas = [{"cenario": "sem_dp", "epsilon": np.nan, "sigma": 0.0,
               "ataque_auc": round(auc0, 4), "ataque_acuracia": round(acc0, 4)}]
    for eps in EPSILONS:
        auc, acc = atacar_com_dp(contextos, CLIP_NORM, eps, DELTA)
        linhas.append({"cenario": f"dp_eps_{eps}", "epsilon": eps,
                       "sigma": round(calcular_sigma(CLIP_NORM, eps, DELTA), 4),
                       "ataque_auc": round(auc, 4), "ataque_acuracia": round(acc, 4)})
        print(f"[DP eps={eps:>4}]  AUC={auc:.4f}  acc={acc:.4f}")

    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(SAIDA, "membership_attack.csv"), index=False)

    com = df[df.cenario != "sem_dp"]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(com.epsilon, com.ataque_auc, marker="o", label="Com DP")
    ax.axhline(auc0, ls="--", color="red", label=f"Sem DP ({auc0:.3f})")
    ax.axhline(0.5, ls=":", color="gray", label="Aleatório (0.5)")
    ax.set_xscale("log"); ax.set_xticks(EPSILONS); ax.set_xticklabels([str(e) for e in EPSILONS])
    ax.set_xlabel("epsilon"); ax.set_ylabel("AUC do ataque")
    _estilo_quadrado(ax)
    fig.savefig(os.path.join(SAIDA, "ataque_auc_vs_epsilon.png"), dpi=120, bbox_inches="tight")
    print("Resultados e gráfico salvos em results/")


if __name__ == "__main__":
    main()
