# Use the official Python image from the Docker Hub
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download models
RUN python -m src.download_models

# Make port 80 available to the world outside this container
EXPOSE 80

# Run uvicorn when the container launches
CMD ["uvicorn", "src.system:app", "--host", "0.0.0.0", "--port", "80"]
