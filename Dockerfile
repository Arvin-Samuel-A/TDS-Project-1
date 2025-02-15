FROM python:3.13

# Install Node.js and npm
RUN apt-get update && \
    apt-get install -y nodejs npm && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install uv
RUN pip install --no-cache-dir uv

# Install the dependencies using uv
RUN uv pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY app.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uv run
CMD ["uv", "run", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]