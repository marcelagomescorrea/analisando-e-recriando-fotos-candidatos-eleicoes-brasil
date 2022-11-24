FROM --platform=linux/amd64 tensorflow/tensorflow:2.10.0

WORKDIR /app

COPY face_rec ./face_rec
COPY fast_api ./fast_api
COPY pca_logic ./pca_logic
COPY autoencoder_logic ./autoencoder_logic
COPY training_outputs ./training_outputs
COPY requirements.txt .

# libraries required by OpenCV
RUN apt-get update
RUN apt-get install \
  'ffmpeg'\
  'libsm6'\
  'libxext6'  -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# You can add --port $PORT if you need to set PORT as a specific env variable
CMD uvicorn fast_api.api:app --host 0.0.0.0 --port $PORT
