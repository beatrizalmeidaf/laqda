FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# instalar Rust (para pacotes como tokenizers)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# criar diretório de trabalho
WORKDIR /app

# copiar requirements
COPY requirements.txt .

# instalar dependências 
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# copiar o projeto
# COPY . .

# CMD ["python", "main.py"]
