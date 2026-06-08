"""Harness de avaliação do trabalho inteiro.

Gera TODOS os resultados do zero a partir apenas dos scripts e do dataset
(client/ai4i2020.csv). Mede, em um único processo:
  - Utilidade do modelo federado (simulação FedAvg de 3 clientes): acurácia global,
    recall e F1 da classe de falha, e a evolução de loss/acurácia por round.
  - Privacidade: eficácia do ataque de inferência de pertencimento (attack/).

Varre uma variável de cada vez (as outras ficam no padrão) para isolar o impacto de
cada uma: epsilon, norma de clip C, delta e número de rodadas. Salva CSVs e gráficos
em results/.
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
from sklearn.metrics import recall_score, f1_score, log_loss

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "client"))
sys.path.insert(0, os.path.join(RAIZ, "attack"))
from dp_utils import aplicar_dp, calcular_sigma
import membership_attack as mia

warnings.filterwarnings("ignore")

DADOS = os.path.join(RAIZ, "client", "ai4i2020.csv")
SAIDA = os.path.join(RAIZ, "results")

FEATURES = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]

EPS_PAD, C_PAD, DELTA_PAD, ROUNDS_PAD = 1.0, 1.0, 1e-5, 20
N_CLIENTES = 3
REPET = 8

EPS_UTIL = [0.5, 1, 2, 4, 8]
EPS_ATAQUE = [0.5, 1, 2, 4, 8, 16, 32, 64]
CLIPS = [0.5, 1, 2]
DELTAS = [1e-3, 1e-5, 1e-7]
ROUNDS = [5, 10, 20]


def carregar_fatias():
    df = pd.read_csv(DADOS)
    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    X = StandardScaler().fit_transform(df[FEATURES].values)
    y = df["Machine failure"].values
    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    return [(X[f], y[f]) for f in np.array_split(idx, N_CLIENTES)]


def novo_modelo(X, y):
    m = LogisticRegression(warm_start=True, max_iter=1, class_weight="balanced")
    m.fit(X, y)
    m.classes_ = np.array([0, 1])
    return m


def fedavg(updates):
    total = sum(n for _, _, n in updates)
    coef = sum(c * n for c, _, n in updates) / total
    intercept = sum(i * n for _, i, n in updates) / total
    return coef, intercept


def metricas_globais(modelos, fatias, coef, intercept, com_loss=False):
    # Junta as predições de todos os clientes e mede acurácia, recall e F1 da falha
    y_true, y_pred, y_prob = [], [], []
    for m, (X, y) in zip(modelos, fatias):
        m.coef_, m.intercept_ = coef, intercept
        y_true.append(y); y_pred.append(m.predict(X))
        if com_loss:
            y_prob.append(m.predict_proba(X))
    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)
    acc = float((yt == yp).mean())
    rec = float(recall_score(yt, yp, pos_label=1, zero_division=0))
    f1 = float(f1_score(yt, yp, pos_label=1, zero_division=0))
    if com_loss:
        loss = float(log_loss(yt, np.concatenate(y_prob), labels=[0, 1]))
        return acc, rec, f1, loss
    return acc, rec, f1


def treino_federado(fatias, epsilon, clip, delta, rounds, rng, historico=False):
    # epsilon=None desliga a DP; senão aplica clipping + ruído a cada rodada.
    # historico=True registra (loss, acurácia) ao final de cada round.
    modelos = [novo_modelo(X, y) for X, y in fatias]

    def parametros(m):
        if epsilon is None:
            return m.coef_, m.intercept_, None
        return aplicar_dp(m.coef_, m.intercept_, clip, epsilon, delta, rng)

    coef, intercept, sigma = parametros(modelos[0])
    hist = []
    for _ in range(rounds):
        updates = []
        for m, (X, y) in zip(modelos, fatias):
            m.coef_, m.intercept_ = coef, intercept
            m.fit(X, y)
            c, i, sigma = parametros(m)
            updates.append((c, i, len(y)))
        coef, intercept = fedavg(updates)
        if historico:
            acc_r, _, _, loss_r = metricas_globais(modelos, fatias, coef, intercept, com_loss=True)
            hist.append((loss_r, acc_r))
    acc, rec, f1 = metricas_globais(modelos, fatias, coef, intercept)
    return acc, rec, f1, sigma, hist


def utilidade(fatias, use_dp, epsilon, clip, delta, rounds, historico=False):
    if not use_dp:
        acc, rec, f1, _, hist = treino_federado(fatias, None, clip, delta, rounds, None, historico)
        return acc, rec, f1, 0.0, (np.array(hist) if historico else None)
    vals, hists, sigma = [], [], None
    for r in range(REPET):
        acc, rec, f1, sigma, hist = treino_federado(fatias, epsilon, clip, delta, rounds,
                                                     np.random.default_rng(1000 + r), historico)
        vals.append((acc, rec, f1))
        if historico:
            hists.append(hist)
    acc, rec, f1 = np.mean(vals, axis=0)
    hist_media = np.mean(hists, axis=0) if historico else None
    return float(acc), float(rec), float(f1), sigma, hist_media


def linha(analise, variavel, use_dp, eps, clip, delta, rounds, sigma,
          acc=np.nan, rec=np.nan, f1=np.nan, auc=np.nan, atk_acc=np.nan):
    n = 1 if analise == "ataque" else rounds
    return {"analise": analise, "variavel": variavel, "use_dp": use_dp,
            "epsilon": eps, "clip_norm": clip, "delta": delta, "num_rounds": rounds,
            "sigma": None if sigma is None else round(sigma, 4),
            "epsilon_total": None if (not use_dp) else round(eps * n, 4),
            "acuracia_global": acc, "recall_falha": rec, "f1_falha": f1,
            "ataque_auc": auc, "ataque_acuracia": atk_acc}


def main():
    os.makedirs(SAIDA, exist_ok=True)
    fatias = carregar_fatias()
    contextos = mia.preparar_alvo()

    linhas = []
    curvas = {}

    acc0, rec0, f10, _, h0 = utilidade(fatias, False, EPS_PAD, C_PAD, DELTA_PAD, ROUNDS_PAD, historico=True)
    curvas["Sem DP"] = h0
    linhas.append(linha("utilidade", "baseline", False, np.nan, np.nan, np.nan, ROUNDS_PAD, 0.0, acc=acc0, rec=rec0, f1=f10))
    auc0, atk0 = mia.atacar_sem_dp(contextos)
    linhas.append(linha("ataque", "baseline", False, np.nan, np.nan, np.nan, ROUNDS_PAD, 0.0, auc=auc0, atk_acc=atk0))
    print(f"Baseline sem DP -> acc={acc0:.4f} recall_falha={rec0:.4f} f1_falha={f10:.4f} | ataque AUC={auc0:.4f}")

    for eps in EPS_UTIL:
        acc, rec, f1, sig, h = utilidade(fatias, True, eps, C_PAD, DELTA_PAD, ROUNDS_PAD, historico=True)
        curvas[f"eps={eps}"] = h
        linhas.append(linha("utilidade", "epsilon", True, eps, C_PAD, DELTA_PAD, ROUNDS_PAD, sig, acc=acc, rec=rec, f1=f1))
    for eps in EPS_ATAQUE:
        auc, atk = mia.atacar_com_dp(contextos, C_PAD, eps, DELTA_PAD)
        linhas.append(linha("ataque", "epsilon", True, eps, C_PAD, DELTA_PAD, ROUNDS_PAD,
                            calcular_sigma(C_PAD, eps, DELTA_PAD), auc=auc, atk_acc=atk))

    for c in CLIPS:
        acc, rec, f1, sig, _ = utilidade(fatias, True, EPS_PAD, c, DELTA_PAD, ROUNDS_PAD)
        linhas.append(linha("utilidade", "clip_norm", True, EPS_PAD, c, DELTA_PAD, ROUNDS_PAD, sig, acc=acc, rec=rec, f1=f1))
        auc, atk = mia.atacar_com_dp(contextos, c, EPS_PAD, DELTA_PAD)
        linhas.append(linha("ataque", "clip_norm", True, EPS_PAD, c, DELTA_PAD, ROUNDS_PAD,
                            calcular_sigma(c, EPS_PAD, DELTA_PAD), auc=auc, atk_acc=atk))

    for d in DELTAS:
        acc, rec, f1, sig, _ = utilidade(fatias, True, EPS_PAD, C_PAD, d, ROUNDS_PAD)
        linhas.append(linha("utilidade", "delta", True, EPS_PAD, C_PAD, d, ROUNDS_PAD, sig, acc=acc, rec=rec, f1=f1))
        auc, atk = mia.atacar_com_dp(contextos, C_PAD, EPS_PAD, d)
        linhas.append(linha("ataque", "delta", True, EPS_PAD, C_PAD, d, ROUNDS_PAD,
                            calcular_sigma(C_PAD, EPS_PAD, d), auc=auc, atk_acc=atk))

    for k in ROUNDS:
        acc, rec, f1, sig, _ = utilidade(fatias, True, EPS_PAD, C_PAD, DELTA_PAD, k)
        linhas.append(linha("utilidade", "num_rounds", True, EPS_PAD, C_PAD, DELTA_PAD, k, sig, acc=acc, rec=rec, f1=f1))

    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(SAIDA, "experiments_master.csv"), index=False)
    print(f"CSV mestre salvo: results/experiments_master.csv ({len(df)} cenários)")
    salvar_convergencia(curvas)
    gerar_graficos(df)
    gerar_curvas(curvas)
    print("CSVs e gráficos salvos em results/")


def _estilo(ax):
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False, fontsize=8)


def _salvar(fig, nome):
    fig.savefig(os.path.join(SAIDA, nome), dpi=120, bbox_inches="tight")
    plt.close(fig)


def salvar_convergencia(curvas):
    rows = []
    for cen, h in curvas.items():
        for r, (loss, acc) in enumerate(h, start=1):
            rows.append({"cenario": cen, "round": r, "loss": round(float(loss), 4), "acuracia": round(float(acc), 4)})
    pd.DataFrame(rows).to_csv(os.path.join(SAIDA, "convergencia.csv"), index=False)


def gerar_graficos(df):
    bu = df[(df.analise == "utilidade") & (df.variavel == "baseline")].iloc[0]
    ba = df[(df.analise == "ataque") & (df.variavel == "baseline")].iloc[0]
    u = df[(df.analise == "utilidade") & (df.variavel == "epsilon")]
    a = df[(df.analise == "ataque") & (df.variavel == "epsilon")]

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(u.epsilon, u.acuracia_global, marker="o", label="Com DP")
    ax.axhline(bu.acuracia_global, ls="--", color="gray", label=f"Sem DP ({bu.acuracia_global:.3f})")
    ax.set_xscale("log"); ax.set_xticks(EPS_UTIL); ax.set_xticklabels([str(e) for e in EPS_UTIL])
    ax.set_xlabel("epsilon"); ax.set_ylabel("Acurácia global")
    _estilo(ax); _salvar(fig, "acuracia_vs_epsilon.png")

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(u.epsilon, u.recall_falha, marker="o", label="Recall (falha)")
    ax.plot(u.epsilon, u.f1_falha, marker="s", label="F1 (falha)")
    ax.axhline(bu.recall_falha, ls="--", color="tab:blue", alpha=0.5, label=f"Recall sem DP ({bu.recall_falha:.3f})")
    ax.axhline(bu.f1_falha, ls="--", color="tab:orange", alpha=0.5, label=f"F1 sem DP ({bu.f1_falha:.3f})")
    ax.set_xscale("log"); ax.set_xticks(EPS_UTIL); ax.set_xticklabels([str(e) for e in EPS_UTIL])
    ax.set_xlabel("epsilon"); ax.set_ylabel("Métrica da classe de falha")
    _estilo(ax); _salvar(fig, "recall_f1_vs_epsilon.png")

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(a.epsilon, a.ataque_auc, marker="o", label="Com DP")
    ax.axhline(ba.ataque_auc, ls="--", color="red", label=f"Sem DP ({ba.ataque_auc:.3f})")
    ax.axhline(0.5, ls=":", color="gray", label="Aleatório (0.5)")
    ax.set_xscale("log"); ax.set_xticks(EPS_ATAQUE); ax.set_xticklabels([str(e) for e in EPS_ATAQUE])
    ax.set_xlabel("epsilon"); ax.set_ylabel("AUC do ataque")
    _estilo(ax); _salvar(fig, "ataque_auc_vs_epsilon.png")

    # Comparação direta: ataque no modelo SEM DP vs COM DP no epsilon padrão (1).
    # O ataque fica em ~0.50 em qualquer epsilon; a ordem de privacidade vem da
    # garantia formal (menor epsilon = mais privado), não desta métrica empírica.
    da = df[(df.analise == "ataque") & (df.variavel == "epsilon") & (df.epsilon == EPS_PAD)].iloc[0]
    rotulos = ["AUC", "Acurácia"]
    sem = [ba.ataque_auc, ba.ataque_acuracia]
    com = [da.ataque_auc, da.ataque_acuracia]
    x = np.arange(len(rotulos)); w = 0.35
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    b1 = ax.bar(x - w / 2, sem, w, label="Sem DP")
    b2 = ax.bar(x + w / 2, com, w, label=f"Com DP (eps={EPS_PAD:g})")
    ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.axhline(0.5, ls=":", color="gray", label="Aleatório (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(rotulos)
    ax.set_ylim(0.45, 0.60); ax.set_ylabel("Eficácia do ataque")
    _estilo(ax); _salvar(fig, "ataque_com_vs_sem_dp.png")


def gerar_curvas(curvas):
    rounds = range(1, len(next(iter(curvas.values()))) + 1)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for cen, h in curvas.items():
        ax.plot(rounds, h[:, 0], marker="o", ms=3, label=cen)
    ax.set_xlabel("Round"); ax.set_ylabel("Loss (log-loss)")
    _estilo(ax); _salvar(fig, "loss_vs_rounds.png")

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for cen, h in curvas.items():
        ax.plot(rounds, h[:, 1], marker="o", ms=3, label=cen)
    ax.set_xlabel("Round"); ax.set_ylabel("Acurácia global")
    _estilo(ax); _salvar(fig, "acuracia_vs_rounds.png")


if __name__ == "__main__":
    main()
