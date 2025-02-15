FROM python:3.13

# Install Node.js and npm
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

    RUN apt-get update && apt-get install -y \
    git \
    curl \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure git (replace with your details)
RUN git config --global user.name "Your Name" \
    && git config --global user.email "your.email@example.com"

ARG GITHUB_TOKEN
ARG AIPROXY_TOKEN
# Set it as an environment variable if provided
ENV AIPROXY_TOKEN=${AIPROXY_TOKEN}

RUN if [ -n "$GITHUB_TOKEN" ]; then \
      echo $GITHUB_TOKEN | gh auth login --with-token; \
    fi

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