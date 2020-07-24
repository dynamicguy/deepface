FROM tensorflow/tensorflow:1.9.0-devel-gpu-py3

COPY requirements.txt .

RUN add-apt-repository ppa:deadsnakes/ppa -y 
RUN apt-get update
RUN apt-get install -y python3.6 
RUN apt install -y python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py --force-reinstall
RUN pip install deepface
RUN pip install -r requirements.txt
RUN apt-get install -y libsm6 libxext6 libxrender-dev

EXPOSE 80

COPY ./deepface ./deepface

RUN mkdir /root/.deepface 
RUN mkdir /root/.deepface/weights
RUN mv deepface/weights/* /root/.deepface/weights/

COPY . .

CMD ["python","api/api.py"]


