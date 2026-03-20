FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

RUN python -m pip install fastapi uvicorn pydantic pandas python-multipart httpx google-genai langchain_openai --no-cache-dir  \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

COPY app /app/app

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]