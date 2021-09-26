FROM tensorflow/tensorflow
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-opencv python3-tk
RUN python3 -m pip install tqdm scikit-image pandas matplotlib scikit-learn progressbar
RUN mkdir /app
COPY app /app
USER 1000:1000
CMD ["python3", "/app/main.py"]
