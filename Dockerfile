FROM python:3.10-slim-buster

WORKDIR /app

COPY flask_api/app.py /app/app.py
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]