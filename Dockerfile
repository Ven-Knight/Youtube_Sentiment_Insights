# Use a lightweight Python 3.10 base image for faster builds
FROM python:3.10-slim-bookworm          

# Set working directory inside the container to /app
WORKDIR /app                           

# Copy all files from your repo into the container /app directory
COPY . /app                            

# Copy environment variables file into the container for secure runtime configuration
COPY flask_api/.env /app/flask_api/.env

# Install system-level dependencies required by LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Install Python dependencies listed in requirements.txt
RUN pip install -r requirements.txt    

# Start the Flask API by running app.py inside flask_api folder
CMD ["python3", "flask_api/app.py"]    
