FROM python:3.10.12-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Command to run when the container starts
# You can override this with docker run command
CMD ["python", "main.py", "-d", "glue", "-ds", "sst2", "-m", "bert-base-uncased"]