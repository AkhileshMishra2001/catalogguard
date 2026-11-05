FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY catalogguard ./catalogguard
RUN useradd -u 10001 -m appuser
USER appuser
ENV PYTHONUNBUFFERED=1 CATALOGGUARD_HOME=/app
EXPOSE 8000
CMD ["uvicorn", "catalogguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
