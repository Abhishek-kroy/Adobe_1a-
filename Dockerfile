# Base Image
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY ./app ./app
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure no internet access required
ENV TRANSFORMERS_OFFLINE=1

# Command to run the script
ENTRYPOINT ["python", "app/main.py"]