FROM python:3.10-slim

WORKDIR /app

# TODO: Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# TODO: Copy requirements files
COPY requirements.txt ./requirements.txt
COPY ../requirements.txt ./base_requirements.txt

# TODO: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r base_requirements.txt

# TODO: Copy application code
COPY . .

# TODO: Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# TODO: Expose port for Flask API
EXPOSE 5000

# TODO: Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=api.py

# TODO: Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# TODO: Run Flask application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "api:app"] 