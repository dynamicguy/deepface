FROM python:3.6

COPY requirements.txt .

RUN pip install deepface
RUN pip install -r requirements.txt

EXPOSE 5000

COPY . .

CMD ["python","api/api.py"]


