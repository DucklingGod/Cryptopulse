FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential gcc

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Copy data and models folders for ML model and data access (these will overwrite any empty/incorrect ones from above)
COPY data/ ./data/
COPY models/ ./models/
COPY models/ ./backend/models/
COPY src/run_daily_pipeline.py ./src/run_daily_pipeline.py

# Debug: list model files
RUN ls -lh ./models/
RUN ls -lh ./data/

# Expose port (Cloud Run uses $PORT)
EXPOSE 8080

# Default entrypoint for Cloud Run Job (can be overridden)
CMD ["python", "src/run_daily_pipeline.py"]