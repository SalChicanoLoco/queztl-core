FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies (including git for Shap-E)
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application code as a package
COPY backend/ backend/

# Expose port
EXPOSE 8000

# Run the application as a Python module
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
