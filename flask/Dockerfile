FROM python:3.9.2

# upgrade pip
RUN pip install --upgrade pip

WORKDIR /app
COPY ./requirements.txt /app
RUN export FLASK_APP=app.py

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

EXPOSE 5000
#ENV FLASK_APP=app.py
CMD ["python", "app.py"]

