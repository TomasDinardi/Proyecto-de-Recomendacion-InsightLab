FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# requirements.txt está en la raíz de MLOps-InsightLab, no en src/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código fuente
COPY src/ ./src

# Si api.py/streamlit_app.py necesitan el CSV, lo copiamos también
COPY Ecommerce.csv .

EXPOSE 8000

# OJO: ahora api.py está en /app/src/api.py, no en /app/api.py
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]