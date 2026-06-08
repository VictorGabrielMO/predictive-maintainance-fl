import os
import time
import warnings

import flwr as fl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from grpc import RpcError

from dp_utils import aplicar_dp


warnings.filterwarnings("ignore")

def main():
    # 1. Carrega o dataset de manutenção preditiva baixado no Dockerfile
    df = pd.read_csv("ai4i2020.csv")

    # Converte a coluna de texto 'Type' (L, M, H) em números (0, 1, 2)
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    df['Type'] = df['Type'].map(type_mapping)

    # Seleciona as colunas de sensores (Features)
    feature_cols = [
        'Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    X = df[feature_cols].values
    y = df['Machine failure'].values

    # Normalização dos dados (fundamental para dados de sensores com escalas muito diferentes)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 2. Simula dados distribuídos usando o CLIENT_ID (0, 1 ou 2)
    client_id = int(os.getenv("CLIENT_ID", 0))

    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    # Divide as 10.000 linhas em 3 fatias (uma para cada fábrica/container)
    fatias = np.array_split(indices, 3)
    meus_indices = fatias[client_id]
    X_local, y_local = X[meus_indices], y[meus_indices]

    # 3. Inicializa o modelo de Regressão Logística
    # Usamos class_weight='balanced' porque falhas de máquinas são eventos raros (dados desbalanceados)
    model = LogisticRegression(warm_start=True, max_iter=1, class_weight='balanced')
    model.fit(X_local, y_local)
    model.classes_ = np.array([0, 1]) # 0 = Normal, 1 = Falha

    # 4. Classe do Cliente Flower dinâmica (Com/Sem Privacidade Diferencial)
    class MaintenanceClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            pesos_reais = model.coef_
            intercept_real = model.intercept_

            # Lê a variável de ambiente (Padrão é False se não for informada)
            use_dp = os.getenv("USE_DP", "False").lower() == "true"

            if use_dp:
                # Parâmetros do mecanismo gaussiano (orçamento de privacidade POR RODADA)
                clip_norm = float(os.getenv("DP_CLIP_NORM", 1.0))  # cota de sensibilidade L2 (C)
                epsilon = float(os.getenv("DP_EPSILON", 1.0))
                delta = float(os.getenv("DP_DELTA", 1e-5))

                # Clip da norma L2 do vetor [coef_, intercept_] e ruído calibrado por epsilon
                coef_dp, intercept_dp, sigma = aplicar_dp(
                    pesos_reais, intercept_real, clip_norm, epsilon, delta
                )
                print(f"[Fábrica {client_id}] Modo DP ATIVO. clip C={clip_norm}, epsilon={epsilon}, delta={delta}, sigma calculado={sigma:.4f}.")
                print(f"[Fábrica {client_id}] Parâmetros ENVIADOS COM Privacidade Diferencial (clipping + ruído gaussiano).")
                return [coef_dp, intercept_dp]

            # Caso contrário, envia os pesos originais normais
            print(f"[Fábrica {client_id}] Parâmetros ENVIADOS SEM Privacidade Diferencial (Modo Normal).")
            return [pesos_reais, intercept_real]

        def set_parameters(self, parameters):
            model.coef_ = parameters[0]
            model.intercept_ = parameters[1]

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.fit(X_local, y_local)
            print(f"[Fábrica {client_id}] Treino local concluído.")
            return self.get_parameters(config={}), len(X_local), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss = log_loss(y_local, model.predict_proba(X_local), labels=[0, 1])
            accuracy = model.score(X_local, y_local)
            print(f"[Fábrica {client_id}] Avaliação -> Acurácia Local: {accuracy:.4f}")
            return float(loss), len(X_local), {"accuracy": float(accuracy)}

    # 5. Conecta ao servidor central
    server_address = os.getenv("SERVER_ADDRESS", "localhost:8080")
    print(f"Iniciando Cliente da Fábrica {client_id} conectado em {server_address}...")

    # Tenta ligar-se até 10 vezes antes de crashar o container
    for tentativa in range(10):
        try:
            fl.client.start_numpy_client(server_address=server_address, client=MaintenanceClient())
            break # Se conseguir conectar com sucesso, sai do loop
        except RpcError:
            print(f"Servidor ainda não está pronto. Tentativa {tentativa+1}/10. Aguardando 3 segundos...")
            time.sleep(3)

if __name__ == "__main__":
    main()
