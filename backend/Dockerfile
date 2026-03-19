FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    openssl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

RUN python -m pip install fastapi uvicorn pydantic pandas python-multipart httpx --no-cache-dir  \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

COPY app /app/app
COPY patients.csv /app/app/patients.csv

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]