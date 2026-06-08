# 🏭 Aprendizado Federado para Manutenção Preditiva na Indústria 4.0

Este projeto implementa um sistema prático de **Aprendizado Federado (Federated Learning)** utilizando o framework **Flower (`flwr`)** e a biblioteca **Scikit-Learn**. O objetivo principal é treinar de forma colaborativa um modelo de Machine Learning capaz de prever falhas em maquinários industriais (Manutenção Preditiva) preservando a privacidade dos dados locais de cada fábrica.

Além disso, o projeto conta com suporte a **Privacidade Diferencial (DP)** via **mecanismo gaussiano**, com o desvio-padrão do ruído **calibrado a partir do orçamento de privacidade `epsilon`**, permitindo avaliar o trade-off prático entre privacidade e acurácia. Um **ataque de inferência de pertencimento (membership inference)** acompanha o projeto para demonstrar o ganho de privacidade.

---

## 📈 Cenário e Dataset

O sistema simula **3 fábricas (Clientes)** independentes que monitorizam os seus próprios sensores industriais. Devido a segredos de mercado, conformidade com leis de proteção de dados ou limitações de infraestrutura, as fábricas não partilham os seus dados brutos com um servidor central.

O conjunto de dados utilizado é o **AI4I 2020 Predictive Maintenance Dataset**. O modelo de Regressão Logística analisa as seguintes características (features) dos sensores:

- **Type**: Tipo do produto (L, M ou H convertido para 0, 1 ou 2)
- **Air temperature [K]**: Temperatura do ar
- **Process temperature [K]**: Temperatura do processo
- **Rotational speed [rpm]**: Velocidade de rotação
- **Torque [Nm]**: Torque aplicado
- **Tool wear [min]**: Desgaste da ferramenta

O objetivo do modelo é classificar a variável alvo **Machine failure** entre `0` (Funcionamento Normal) ou `1` (Falha Mecânica).

---

## 🔐 Privacidade Diferencial (mecanismo gaussiano)

Quando `USE_DP=True`, cada fábrica protege os parâmetros do seu modelo (`coef_` e `intercept_`) **antes de os enviar ao servidor**, em duas etapas:

**1. Clipping da sensibilidade.** Os parâmetros são concatenados num único vetor e a sua **norma L2** é limitada a um valor máximo `C` (`DP_CLIP_NORM`). Se a norma for maior que `C`, o vetor é reescalado para `C`; caso contrário, é mantido. Esse `C` é a **cota de sensibilidade L2** (`Δ₂`) do mecanismo.

**2. Ruído gaussiano calibrado.** Adiciona-se ruído `N(0, σ²)` a **todos** os componentes do vetor, com o desvio-padrão **calculado** pela fórmula do mecanismo gaussiano (slides da disciplina):

```
σ = (C / epsilon) · √( ln(1 / delta) )
```

> ⚠️ **A relação correta:** o que se varia é o **`epsilon`**, e o desvio-padrão `σ` é **calculado** a partir dele — não é informado direto.
> **`epsilon` MENOR ⇒ `σ` MAIOR ⇒ MAIS privacidade ⇒ MENOS acurácia.**

O mecanismo satisfaz **(ε, δ)-DP** por rodada de comunicação.

### Composição: o orçamento se acumula entre rodadas

O treino federado tem várias rodadas (`NUM_ROUNDS`, padrão **20**), e cada rodada envia uma versão ruidosa dos parâmetros. Pela **composição**, o orçamento de privacidade **se acumula**: o `DP_EPSILON` configurado é o orçamento **POR RODADA**, não do treino inteiro.

- **Composição simples (cota superior):** `epsilon_total ≤ NUM_ROUNDS × DP_EPSILON` (e `delta_total ≤ NUM_ROUNDS × DP_DELTA`). Ex.: `DP_EPSILON=1` e 20 rodadas ⇒ `epsilon_total ≤ 20`.
- **Composição avançada:** para `epsilon` pequeno, o total cresce aproximadamente proporcional a `√(NUM_ROUNDS)`, podendo ser mais justo que a soma simples.

Por isso **não** se deve afirmar um `epsilon` pequeno único para todo o treino. As tabelas de resultados reportam o `epsilon` por rodada e o total aproximado pela composição simples.

### Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `USE_DP` | `True` | Liga/desliga a Privacidade Diferencial |
| `DP_EPSILON` | `1.0` | Orçamento de privacidade **por rodada** (menor = mais privacidade) |
| `DP_DELTA` | `1e-5` | Fator de relaxação δ do mecanismo gaussiano |
| `DP_CLIP_NORM` | `1.0` | Norma L2 máxima `C` (cota de sensibilidade) |
| `NUM_ROUNDS` | `20` | Número de rodadas de comunicação |

---

## 🛠️ Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) e [Docker Compose](https://docs.docker.com/compose/install/)
- Para os scripts de análise sem Docker: Python 3.10+ com `scikit-learn`, `pandas`, `numpy`, `matplotlib`.

---

## 🚀 Como Executar (Docker)

Toda a rede interna do Docker e a instalação de dependências ocorrem automaticamente ao subir os containers.

### Modo 1: COM Privacidade Diferencial

O sistema adiciona o ruído gaussiano calibrado pelo `epsilon` aos parâmetros de cada fábrica antes de enviá-los ao servidor. Para escolher o orçamento de privacidade por rodada:

```bash
DP_EPSILON=<valor> docker compose up --build
```

- Substitua `<valor>` pelo orçamento por rodada (ex.: `0.5`, `1`, `2`, `4`, `8`). **Quanto menor, maior a privacidade e menor a acurácia.**
- Opcionalmente: `DP_CLIP_NORM=<C>`, `DP_DELTA=<δ>`, `NUM_ROUNDS=<k>`.
- `--build` força a reconstrução das imagens (use após alterar o código).

### Modo 2: SEM Privacidade Diferencial

Os clientes enviam os parâmetros originais (limpos) para o servidor agregar via FedAvg:

```bash
USE_DP=False docker compose up --build
```

### Modo 3: Rodar os experimentos (utilidade + ataque)

O harness `run_experiments.py` também roda via Docker, sem precisar instalar Python. Os resultados (CSV + gráficos) aparecem na pasta `results/` da máquina:

```bash
docker compose --profile experiments run --rm experiments
```

Esse serviço tem um *profile* próprio, então **não** sobe junto com o `docker compose up` dos Modos 1 e 2 — só roda quando chamado explicitamente.

### Como Parar

```bash
docker compose down
```

---

## 📊 Experimentos e Análise

O harness `run_experiments.py` (na raiz) avalia **todo o trabalho** num único processo, reutilizando a mesma lógica de DP do cliente (`client/dp_utils.py`). Para cada cenário mede a **utilidade** (do modelo federado, simulação FedAvg de 3 clientes: acurácia global + **recall e F1 da classe de falha** + **evolução de loss/acurácia por round**) e a **privacidade** (eficácia do ataque de pertencimento). Varre **uma variável de cada vez** — as outras ficam no padrão `epsilon=1, C=1, delta=1e-5, rounds=20` — para isolar o impacto de cada uma (epsilon, norma de clip `C`, `delta` e número de rodadas). Basta ter os scripts e o `client/ai4i2020.csv` — **todo o resto (CSVs e gráficos) é gerado do zero**.

Duas formas de rodar — **via Docker** (recomendado para o grupo, não precisa instalar Python) ou **local**:

```bash
# Via Docker — resultados aparecem em ./results
docker compose --profile experiments run --rm experiments

# Ou local (precisa de Python 3.10+)
pip install scikit-learn pandas numpy matplotlib
python run_experiments.py            # roda tudo (utilidade + ataque), gera results/
python attack/membership_attack.py   # opcional: só o ataque, isolado
```

Saídas em `results/` (todas geradas do zero): **`experiments_master.csv`** (todos os cenários: com/sem DP, com/sem ataque, cada varredura), **`convergencia.csv`** (loss e acurácia por round) e os gráficos quadrados **`acuracia_vs_epsilon.png`**, **`recall_f1_vs_epsilon.png`**, **`ataque_auc_vs_epsilon.png`**, **`ataque_com_vs_sem_dp.png`**, **`loss_vs_rounds.png`** e **`acuracia_vs_rounds.png`**. As varreduras de `C`, `delta` e `num_rounds` têm efeito pequeno (no `epsilon` padrão) e ficam só no CSV — melhor representadas em tabela.

### Configuração de cada análise (referência para o relatório)

**Fixo em todos os experimentos:** 3 clientes federados com FedAvg, modelo de Regressão Logística (6 features, `class_weight='balanced'`). Cada ponto de **utilidade com DP** é a **média de 8 execuções** (sementes de ruído diferentes). O **ataque** é o **LiRA**, com alvo no modelo de **um cliente** (100 amostras de treino), 64 modelos de referência e 5 separações.

**Valores padrão** (usados quando a variável não está sendo variada): `epsilon=1`, `C=1`, `delta=1e-5`, `rounds=20`. **Cada figura varia uma única variável** e mantém as outras nos padrões:

| Figura / dado | Análise (tem ataque?) | Variável variada | Outras (fixas) |
|---|---|---|---|
| `acuracia_vs_epsilon.png` | Utilidade (sem ataque) | `epsilon` ∈ {0.5,1,2,4,8} | C=1, δ=1e-5, rounds=20 |
| `recall_f1_vs_epsilon.png` | Utilidade (sem ataque) | `epsilon` ∈ {0.5,1,2,4,8} | C=1, δ=1e-5, rounds=20 |
| `loss_vs_rounds.png` e `acuracia_vs_rounds.png` | Utilidade (sem ataque) | o `round` (1→20); 1 curva por cenário (sem DP + cada `epsilon`) | C=1, δ=1e-5, rounds=20 |
| `ataque_auc_vs_epsilon.png` | Ataque (LiRA) | `epsilon` ∈ {0.5,1,2,4,8,16,32,64} | C=1, δ=1e-5 |
| `ataque_com_vs_sem_dp.png` | Ataque (LiRA) | sem DP × com DP | `epsilon=1` (padrão), C=1, δ=1e-5 |
| CSV — varredura de C | Utilidade + Ataque | `C` ∈ {0.5,1,2} | epsilon=1, δ=1e-5, rounds=20 |
| CSV — varredura de δ | Utilidade + Ataque | `delta` ∈ {1e-3,1e-5,1e-7} | epsilon=1, C=1, rounds=20 |
| CSV — varredura de rounds | Utilidade (sem ataque) | `rounds` ∈ {5,10,20} | epsilon=1, C=1, δ=1e-5 |

Notas:
- **Utilidade e ataque são duas análises separadas sobre a mesma configuração de DP**: a utilidade mede o desempenho do modelo federado (acurácia / recall / F1); o ataque mede se um adversário consegue inferir pertencimento (AUC do ataque).
- A barra "com DP" do `ataque_com_vs_sem_dp.png` usa o `epsilon` **padrão (1)**. O ataque cai para ~0.50 em **todos** os epsilons (faixa 0.504–0.511, dentro do ruído da medição): a DP neutraliza o ataque em qualquer configuração, e essas diferencinhas **não** indicam ordem de privacidade. A ordem real é dada pela garantia formal (ε,δ): **menor `epsilon` = mais privado** (ver seção de composição) — o ataque empírico está saturado no acaso e não captura essa ordem.
- O `rounds` não se aplica ao ataque (ele age no modelo final liberado, uma vez só).
- O **`experiments_master.csv`** traz a configuração completa de cada linha nas colunas `analise`, `variavel`, `epsilon`, `clip_norm`, `delta`, `num_rounds` — é a referência definitiva de cada cenário.

### 1. Utilidade — acurácia, recall e F1 vs. epsilon

Como o dataset é desbalanceado (≈3,4% de falhas), a acurácia crua engana: prever sempre "sem falha" já daria ~0,966. Por isso reportamos também **recall** e **F1 da classe de falha** (o modelo usa `class_weight='balanced'`, priorizando capturar falhas). Variando o `epsilon` por rodada (`C=1, delta=1e-5, 20 rodadas`):

| Cenário | epsilon/rodada | σ | epsilon total | Acurácia | Recall (falha) | F1 (falha) |
|---|---|---|---|---|---|---|
| Sem DP | — | — | — | **0.818** | **0.811** | **0.232** |
| DP | 0.5 | 6.79 | ≤ 10 | 0.537 | 0.478 | 0.070 |
| DP | 1 | 3.39 | ≤ 20 | 0.530 | 0.502 | 0.072 |
| DP | 2 | 1.70 | ≤ 40 | 0.526 | 0.492 | 0.070 |
| DP | 4 | 0.85 | ≤ 80 | 0.557 | 0.653 | 0.095 |
| DP | 8 | 0.42 | ≤ 160 | 0.661 | 0.775 | 0.141 |

Sem DP o modelo captura **81% das falhas** (recall 0,811), com F1 baixo (0,232) por gerar muitos falsos positivos — comportamento esperado de um classificador balanceado numa classe rara. Com mais privacidade (menos `epsilon`) recall e F1 caem rumo ao acaso; com `epsilon` maior, recuperam-se em direção ao sem-DP. Gráficos: `results/acuracia_vs_epsilon.png` e `results/recall_f1_vs_epsilon.png`.

### 2. Privacidade — ataque de inferência de pertencimento (LiRA)

O ataque usado é o **LiRA** (*Likelihood Ratio Attack*, Carlini et al., 2022), a versão **calibrada e estado-da-arte** — bem mais forte que o limiar de confiança ingênuo. Em vez de olhar só a confiança crua (que confunde exemplo fácil com membro), o LiRA treina vários **modelos de referência** que nunca viram o exemplo, para estimar a perda "normal" dele quando NÃO é membro; se o modelo alvo dá uma perda **anormalmente baixa**, o exemplo provavelmente estava no treino. Reporta **AUC** e a melhor **acurácia** balanceada (baseline aleatório = 0.5).

**Alvo:** o modelo **do cliente** — o alvo mais realista para a DP *local* (cada cliente protege a própria contribuição, e um cliente com poucos dados é o mais suscetível). O modelo é a **mesma regressão logística do projeto (6 features), sem nenhuma alteração**.

| Cenário | epsilon | σ | AUC do ataque | Acurácia do ataque |
|---|---|---|---|---|
| **Sem DP** | — | — | **0.528** | **0.554** |
| DP | 0.5 | 6.79 | 0.511 | 0.546 |
| DP | 1 | 3.39 | 0.508 | 0.545 |
| DP | 2 | 1.70 | 0.506 | 0.543 |
| DP | 4 | 0.85 | 0.505 | 0.544 |
| DP | 8 | 0.42 | 0.504 | 0.544 |
| DP | 16 | 0.21 | 0.505 | 0.542 |
| DP | 32 | 0.11 | 0.506 | 0.540 |
| DP | 64 | 0.05 | 0.506 | 0.540 |

**Resultado honesto:** mesmo o melhor ataque (LiRA) extrai pouco — AUC ≈ 0.53 sem DP — porque a regressão logística com 6 features **não decora** esses dados estruturados de sensores (acurácia de treino ≈ acurácia de teste). O vazamento de pertencimento exige overfitting, que esse modelo evita: é um modelo naturalmente robusto. Ainda assim, **com DP a eficácia cai para ≈ 0.50 (aleatório)** em todos os `epsilon`. A curva fica plana porque, com `C=1`, o clipping já remove quase todo o (pequeno) sinal antes mesmo do ruído. A comparação direta **sem DP vs com DP** (no `epsilon` padrão = 1) está em `results/ataque_com_vs_sem_dp.png`: a AUC do ataque cai de **0.528 para 0.508** (≈ acaso) e a acurácia de **0.554 para 0.545**. A variação por epsilon está em `results/ataque_auc_vs_epsilon.png` — o ataque fica em ~0.50 em todos os epsilons; a ordem de privacidade vem da garantia formal (menor epsilon = mais privado), não desta métrica empírica.

### 3. Convergência — loss e acurácia por round

Evolução do modelo global ao longo dos 20 rounds (média sobre as repetições de ruído), para o cenário sem DP e cada `epsilon`. **Sem DP o treino converge suavemente** (loss ≈ 0.40, acurácia ≈ 0.82); quanto **menor o `epsilon`, maior o ruído por rodada**, e mais alta e instável fica a curva — `epsilon=0.5` chega a oscilar com loss entre 3 e 5. Mostra na prática o custo da DP rigorosa na dinâmica de treino. Gráficos: `results/loss_vs_rounds.png` e `results/acuracia_vs_rounds.png`; dados completos em `results/convergencia.csv`.

---

## 👀 O que observar durante a execução (Docker)

Ao subir os containers, acompanhe as mensagens no terminal:

**No Modo Normal:**
```text
[Fábrica X] Parâmetros ENVIADOS SEM Privacidade Diferencial (Modo Normal).
```

**No Modo Protegido** (note o `sigma` calculado a partir do `epsilon`):
```text
[Fábrica X] Modo DP ATIVO. clip C=1.0, epsilon=1.0, delta=1e-05, sigma calculado=3.3931.
[Fábrica X] Parâmetros ENVIADOS COM Privacidade Diferencial (clipping + ruído gaussiano).
```

O servidor imprime a **acurácia global agregada** a cada uma das 20 rodadas. Note como, com DP, o modelo ganha proteção contra fuga de informação ao custo de uma redução de acurácia (o clássico dilema **Acurácia vs. Privacidade**).

---

## 📂 Estrutura de Pastas do Projeto

```text
predictive-maintainance-fl/
├── docker-compose.yml          # Orquestração: servidor + 3 clientes + serviço de experimentos
├── experiments.Dockerfile      # Imagem para rodar o harness via Docker (Modo 3)
├── README.md                   # Este guia
├── run_experiments.py          # Harness: avalia tudo (utilidade + ataque) e gera results/
├── server/
│   ├── Dockerfile
│   └── server.py               # FedAvg + agregação da acurácia global + NUM_ROUNDS
├── client/                     # O alvo: cliente federado (entra na imagem Docker)
│   ├── Dockerfile
│   ├── ai4i2020.csv            # Dataset
│   ├── dp_utils.py             # Clipping L2 + sigma do mecanismo gaussiano + ruído
│   └── client.py               # Cliente federado (DP rigorosa no get_parameters)
├── attack/                     # O ataque, separado do cliente (não entra no Docker)
│   └── membership_attack.py    # Ataque de pertencimento (importável e standalone)
└── results/                    # CSV mestre e gráficos gerados pelo harness
```

> **Nota sobre o Docker:** o sistema federado (Modos 1 e 2) usa só a pasta `client/`. O harness de experimentos (Modo 3) tem a sua própria imagem (`experiments.Dockerfile`), que copia `dp_utils.py`, o CSV, `attack/membership_attack.py` e `run_experiments.py`. Assim, os três modos rodam por Docker e a reorganização em pastas não quebra nada.
