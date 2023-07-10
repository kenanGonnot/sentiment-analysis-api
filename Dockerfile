FROM python:3.9-slim

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

COPY out/ /models/sentiment_analysis/1/

CMD [ "python", "app.py" ]
#CMD [ "python", "wait_for_tfserver.py" ]

EXPOSE 5000