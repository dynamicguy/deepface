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

HEALTHCHECK --interval=3m --timeout=300s CMD sh -c "if [ ! -f /tmp/health.txt ]; then touch /tmp/health.txt && python api/initRequest.py || exit 0 ; else echo \"initRequest.py already executed\"; fi"
# HEALTHCHECK --interval=3m --timeout=300s CMD python api/initRequest.py || exit 0

ENV CUDA_VISIBLE_DEVICES "0"

CMD ["python", "api/api.py"]


