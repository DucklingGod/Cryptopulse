FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential gcc

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy data and models folders for ML model and data access
COPY data/ ./data/
COPY models/ ./models/

# Copy the rest of your code
COPY . .

# Expose port (Cloud Run uses $PORT)
EXPOSE 8080

# Start the app
CMD exec gunicorn -b :$PORT src.main:app