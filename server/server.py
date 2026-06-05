import flwr as fl

def main():
    # Define a estratégia de agregação (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=3,      # Número mínimo de clientes para treinar
        min_evaluate_clients=3, # Número mínimo de clientes para avaliar
        min_available_clients=3 # Aguarda 3 clientes estarem conectados para iniciar
    )
    
    print("Iniciando o Servidor de Aprendizado Federado...")
    # Inicia o servidor Flower escutando em todas as interfaces na porta 8080
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20), # Executa por 5 rodadas de treino
        strategy=strategy,
    )

if __name__ == "__main__":
    main()