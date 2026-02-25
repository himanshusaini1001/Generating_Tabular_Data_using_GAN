# Use a slim Python image (CPU)
FROM python:3.10-slim

# Avoid Python buffering and .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps (you can add BLAS, etc. later if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port Hugging Face expects
EXPOSE 7860

# Start the Flask app with gunicorn on port 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
