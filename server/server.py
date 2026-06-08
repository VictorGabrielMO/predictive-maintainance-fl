import os
import flwr as fl


def media_ponderada_acuracia(metrics):
    # Agrega a acurácia dos clientes pela média ponderada no número de amostras
    total = sum(num for num, _ in metrics)
    soma = sum(num * m["accuracy"] for num, m in metrics)
    return {"accuracy": soma / total}


def main():
    # Define a estratégia de agregação (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=3,      # Número mínimo de clientes para treinar
        min_evaluate_clients=3, # Número mínimo de clientes para avaliar
        min_available_clients=3, # Aguarda 3 clientes estarem conectados para iniciar
        evaluate_metrics_aggregation_fn=media_ponderada_acuracia, # Acurácia global agregada
    )

    # Número de rodadas configurável (o orçamento de DP é contado POR rodada)
    num_rounds = int(os.getenv("NUM_ROUNDS", 20))

    print("Iniciando o Servidor de Aprendizado Federado...")
    # Inicia o servidor Flower escutando em todas as interfaces na porta 8080
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
