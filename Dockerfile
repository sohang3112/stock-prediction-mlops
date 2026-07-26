FROM python:3.14  
RUN apt-get update && apt-get install -y openjdk-21-jdk && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# TODO: limit to only necessary files like *.py (many unnecessary files from repo also being added to docker image!)
COPY . .

CMD ["fastapi", "run", "prediction_server.py"]

# Spark runs on Java 17/21, Scala 2.13, Python 3.9+