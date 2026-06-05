# 🏭 Aprendizado Federado para Manutenção Preditiva na Indústria 4.0

Este projeto implementa um sistema prático de **Aprendizado Federado (Federated Learning)** utilizando o framework **Flower (`flwr`)** e a biblioteca **Scikit-Learn**. O objetivo principal é treinar de forma colaborativa um modelo de Machine Learning capaz de prever falhas em maquinários industriais (Manutenção Preditiva) preservando a privacidade dos dados locais de cada fábrica.

Além disso, o projeto conta com suporte integrado para **Privacidade Diferencial Local (LDP)**, permitindo avaliar em tempo real o trade-off prático entre a segurança dos dados e a acurácia global do modelo.

---

## 📈 Cenário e Dataset

O sistema simula **3 fábricas (Clientes)** independentes que monitorizam os seus próprios sensores industriais. Devido a segredos de mercado, conformidade com leis de proteção de dados ou limitações de infraestrutura (latência e volume massivo de dados), as fábricas não partilham os seus dados brutos com um servidor central.

O conjunto de dados utilizado é o **AI4I 2020 Predictive Maintenance Dataset** (descarregado automaticamente do repositório público UCI). O modelo de Regressão Logística analisa as seguintes características (features) dos sensores:

- **Type**: Tipo do produto (L, M ou H convertido para 0, 1 ou 2)
- **Air temperature [K]**: Temperatura do ar
- **Process temperature [K]**: Temperatura do processo
- **Rotational speed [rpm]**: Velocidade de rotação
- **Torque [Nm]**: Torque aplicado
- **Tool wear [min]**: Desgaste da ferramenta

O objetivo do modelo é classificar a variável alvo **Machine failure** entre `0` (Funcionamento Normal) ou `1` (Falha Mecânica).

---

## 🛠️ Pré-requisitos

Antes de começar, garante que tens instalado na tua máquina:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (já integrado nativamente nas versões modernas do Docker)

---

## 🚀 Como Executar o Projeto

O projeto foi totalmente dockerizado. Toda a rede interna do Docker, a instalação de dependências e o download do dataset ocorrem de forma 100% automática ao subir os containers.

### 1. Aceder à Pasta do Projeto

Abre o terminal e navega até à raiz do diretório onde se encontram os teus ficheiros de configuração:

```bash
cd meu_projeto_fl/
```


### 2. Modo 1: Executar COM Privacidade Diferencial (Modo Protegido)

Neste modo, o sistema adiciona um ruído estatístico Gaussiano matematicamente controlado aos parâmetros do modelo de cada fábrica antes de os enviar para o servidor, protegendo o ecossistema contra ataques de engenharia reversa e inferência de dados.

Para ativar esta camada de proteção via variável de ambiente, executa:

```bash
DP_NOISE_SCALE=<epsilon> docker compose up [--build]
```
Obs: 
- Substitui `<epsilon>` por um valor numérico que controla a intensidade do ruído (exemplo: `0.05`). Quanto menor o valor, maior a privacidade, mas menor a acurácia.
- O parâmetro `--build` é opcional e força a reconstrução das imagens Docker, útil se tiver feito alterações no código.    

---

### 3. Modo 2: Executar SEM Privacidade Diferencial (Modo Normal)

Neste modo, os clientes treinam os modelos locais e enviam os parâmetros originais (limpos) para o servidor central agregar através do algoritmo FedAvg (Federated Averaging).

No terminal, executa:

```bash
USE_DP=False docker compose up [--build] 
```

## 📊 O que observar durante a execução?

Assim que os containers iniciarem, acompanha as mensagens exibidas no terminal para analisar o comportamento do sistema:

### Orquestração Inicial

O container `flwr_server` iniciará e ficará em modo de escuta. Logo a seguir, os containers de clientes arrancam, dividem o dataset de forma dinâmica e automática e ligam-se ao servidor.

### Rodadas de Comunicação (Rounds 1 a 5)

Acompanha as 5 rodadas de treino federado. Confirma as mensagens dinâmicas:

**No Modo Normal:**

```text
[Fábrica X] Parâmetros ENVIADOS SEM Privacidade Diferencial (Modo Normal).
```

**No Modo Protegido:**

```text
[Fábrica X] Parâmetros ENVIADOS COM Privacidade Diferencial (Ruído adicionado).
```

### Análise de Acurácia

Observa a evolução do indicador **Acurácia Local** impresso por cada fábrica a cada round. Nota como no Modo 2 (com DP) o modelo ganha imunidade contra fugas de informação, mas a acurácia final pode sofrer uma ligeira redução devido ao ruído introduzido (o clássico dilema de **Acurácia vs. Privacidade**).

---

## ⏹️ Como Parar o Sistema

Para encerrar todos os containers e limpar os recursos criados na rede interna isolada do Docker, pressiona `Ctrl + C` no terminal onde o compose está a rodar e executa o comando de limpeza:

```bash
docker compose down
```

---

## 📂 Estrutura de Pastas do Projeto

```text
meu_projeto_fl/
├── docker-compose.yml     # Orquestração dos containers (Servidor e Réplicas do Cliente)
├── README.md              # Este guia de instruções e documentação
├── server/
│   ├── Dockerfile         # Configuração do ambiente isolado do servidor Flower
│   └── server.py          # Lógica de inicialização e estratégia de agregação
└── client/
    ├── Dockerfile         # Configuração do ambiente com Scikit-Learn, Pandas e download do dataset
    ├── ai4i2020.csv       # Dataset 
    └── client.py          # Lógica do cliente, processamento do CSV e injeção dinâmica de ruído DP
```