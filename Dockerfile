# Use a slim Python image
FROM python:3.11-slim

# Create & set app directory
WORKDIR /app

# Copy only requirements, install deps (speeds up rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the port Fly expects (8080 by default)
EXPOSE 8080

# Start your FastAPI app via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]