# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Ensure the entire dataset directory is copied
COPY dataset /app/dataset

# Install Poetry
RUN pip install poetry

# Install dependencies
RUN poetry install

# Expose the port the app runs on
EXPOSE 5001

# Set the command to run the application
CMD ["poetry", "run", "python", "main.py"]