# Use a smaller base image with Python 3.12
FROM python:3.12-slim AS base

# Set the working directory
WORKDIR /app

# Copy only the requirements file to the working directory
COPY requirements.txt .

# Upgrade pip without using cache and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Build stage
FROM base AS build

# Copy the rest of the application code
COPY . .

# Add any other build steps as needed

# Final stage
FROM base AS final

# Set the working directory
WORKDIR /app

# Copy both the application code and installed dependencies from the build stage
COPY --from=build /app /app

# Specify the command to run your application
CMD ["python", "webapp.py"]