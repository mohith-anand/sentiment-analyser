# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy API requirements and install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy all project files
COPY . .

# Expose port 8000 for the FastAPI app (Render will set $PORT)
EXPOSE 8000

# Run the FastAPI app with uvicorn. Use $PORT if provided by the host.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]

