# Imagem para rodar o harness de experimentos (utilidade + ataque) via Docker.
# Não faz parte do sistema federado; é executada sob demanda e grava em results/.
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir scikit-learn numpy pandas matplotlib

# Reproduz a estrutura que run_experiments.py espera (client/ e attack/)
COPY client/dp_utils.py ./client/dp_utils.py
COPY client/ai4i2020.csv ./client/ai4i2020.csv
COPY attack/membership_attack.py ./attack/membership_attack.py
COPY run_experiments.py ./run_experiments.py

CMD ["python", "run_experiments.py"]
