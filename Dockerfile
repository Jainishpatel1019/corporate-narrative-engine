FROM python:3.11-slim

WORKDIR /app

# Copy requirement and install explicitly to leverage layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and data
COPY src/ src/
COPY app.py .
COPY data/ data/

# Expose API port
EXPOSE 8000

# Start the FastAPI service
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
