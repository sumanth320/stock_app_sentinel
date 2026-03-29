# Step 1: Use a lightweight Python base
FROM python:3.11-slim

# Step 2: Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Step 3: Create a working directory
WORKDIR /app

COPY requirements.txt .

# 2. Use BuildKit Cache to keep libraries on your Mac between builds
# This ensures that if only 1 version changes, only 1 library is redownloaded.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 3. Copy the rest of the code (app.py, etc.)
COPY . .

CMD ["python", "app.py"]