# Use a lightweight Python 3.10 base image for faster builds
FROM python:3.10-slim-buster           

# Set working directory inside the container to /app
WORKDIR /app                           

# Copy all files from your repo into the container /app directory
COPY . /app                            

# Install Python dependencies listed in requirements.txt
RUN pip install -r requirements.txt    

# Start the Flask API by running app.py inside flask_api folder
CMD ["python3", "flask_api/app.py"]    
