FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir flwr==1.4.0

COPY server.py .

EXPOSE 8080

CMD ["python", "server.py"]