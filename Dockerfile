FROM python:3.14  
RUN apt-get update && apt-get install -y openjdk-21-jdk && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

CMD ["fastapi", "run"]

# Spark runs on Java 17/21, Scala 2.13, Python 3.9+