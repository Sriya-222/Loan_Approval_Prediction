FROM python:3.11-slim

# Install system utilities if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose FastAPI and Streamlit ports internally
EXPOSE 8000
EXPOSE 8501

# Grant execution permissions to start script
RUN chmod +x start.sh

# Run startup script
CMD ["sh", "start.sh"]
