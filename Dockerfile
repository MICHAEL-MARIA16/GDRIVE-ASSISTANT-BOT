# Use Python 3.10 slim image for better performance
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=chatbot_api.py
ENV KB_PATH=/app/chatbotKB_test

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip first
RUN pip install --upgrade pip

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements files
COPY requirements*.txt ./

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --timeout=300 -r requirements.docker.txt || \
    (echo "Failed to install base requirements" && cat requirements.docker.txt && exit 1)

# Only install Windows requirements if the file exists and has content
#RUN if [ -s requirements.windows.txt ]; then \
        #ip install --no-cache-dir --timeout=300 -r requirements.windows.txt || \
        #(echo "Failed to install Windows requirements" && cat requirements.windows.txt && exit 1); \
    #fi

# Copy application files
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs && \
    touch /app/file_index.db && \
    chown -R appuser:appuser /app

RUN mkdir -p /app/logs /app/chatbotKB_test && \
    touch /app/file_index.db && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "chatbot_api.py"]