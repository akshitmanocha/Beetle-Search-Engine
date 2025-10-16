# Use Python 3.13 slim image for minimal size
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY static/ ./static/
COPY app.py .
COPY main.py .
COPY params.yaml .

# Create data directory
RUN mkdir -p /app/data

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["python", "app.py"]