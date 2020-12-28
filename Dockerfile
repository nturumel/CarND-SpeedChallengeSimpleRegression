FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install --upgrade pip
RUN python -m  pip install moviepy
RUN python -m  pip install scipy
RUN python -m  pip install opencv-python
RUN python -m  pip install keras
RUN python -m pip install scikit-learn