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
                # Se DP estiver ativo, adiciona o ruído Gaussiano
                escala_do_ruido = float(os.getenv("DP_NOISE_SCALE", 0.05)) # Permite configurar a escala do ruído via variável de ambiente
                print (f"[Fábrica {client_id}] Modo DP ATIVO. Adicionando ruído gaussiano com escala {escala_do_ruido} aos pesos.")
                ruido_pesos = np.random.normal(0, escala_do_ruido, pesos_reais.shape)
                pesos_finais = pesos_reais + ruido_pesos
                print(f"[Fábrica {client_id}] Parâmetros ENVIADOS COM Privacidade Diferencial (Ruído adicionado).")
            else:
                # Caso contrário, envia os pesos originais normais
                pesos_finais = pesos_reais
                print(f"[Fábrica {client_id}] Parâmetros ENVIADOS SEM Privacidade Diferencial (Modo Normal).")
            
            return [pesos_finais, intercept_real]

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