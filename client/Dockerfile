FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir flwr==1.4.0 scikit-learn numpy pandas

COPY client.py .
COPY ai4i2020.csv .

CMD ["python", "client.py"]