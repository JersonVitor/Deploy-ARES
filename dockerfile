FROM python:3.10-slim

# Configurações básicas
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Instale dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante da aplicação
COPY . .

# Comando de inicialização da FastAPI
CMD ["uvicorn", "src.python.main:app", "--host=0.0.0.0", "--port=8000"]
