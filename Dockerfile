FROM python:3.10-slim-buster           # Use a lightweight Python 3.10 base image for faster builds

WORKDIR /app                           # Set working directory inside the container to /app

COPY . /app                            # Copy all files from your repo into the container /app directory

RUN pip install -r requirements.txt    # Install Python dependencies listed in requirements.txt

CMD ["python3", "flask_api/app.py"]    # Start the Flask API by running app.py inside flask_api folder
