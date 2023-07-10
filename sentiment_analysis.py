import os
import sys
import requests
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, BertConfig

tokenizer = AutoTokenizer.from_pretrained("/models/sentiment_analysis/1/")
config = BertConfig.from_pretrained("/models/sentiment_analysis/1/config.json")
model = TFAutoModelForSequenceClassification.from_pretrained("/models/sentiment_analysis/1/tf_model.h5", config=config)


def predict_sentiment_basic(sentence: str):
    """Prediction of sentiment analysis (basic)

    Args:
        sentence (string): Sentence to analyse the sentiment 

    Returns:
        sentiment: 'POSITIVE' or 'NEGATIVE'
        confidence: float(range(0, 1))
    """
    inputs = tokenizer(sentence, return_tensors='tf')

    outputs = model(inputs)

    logits = outputs.logits.numpy()[0]
    sentiment = 'NEGATIVE' if logits[0] > logits[1] else 'POSITIVE'
    confidence = tf.nn.softmax(logits).numpy()[0 if sentiment == 'NEGATIVE' else 1]
    return sentiment, confidence


def predict_sentiment_tf_serving(sentence: str):
    """Get prediction from tf-serving server
    Args:
        sentence (string): Sentence to analyse the sentiment

    Returns:
        sentiment: 'POSITIVE' or 'NEGATIVE'
        confidence: float(range(0, 1))
    """
    tokenized_sentence = tokenizer(sentence, return_tensors='tf', max_length=512, truncation=True)
    input_ids = tokenized_sentence['input_ids'].numpy().tolist()[0]
    attention_mask = tokenized_sentence['attention_mask'].numpy().tolist()[0]
    token_type_ids = tokenized_sentence['token_type_ids'].numpy().tolist()[0]

    request = {
        "instances": [{"input_ids": input_ids,
                       "attention_mask": attention_mask,
                       "token_type_ids": token_type_ids}]}
    headers = {"content-type": "application/json"}
    url = os.environ['TF_SERVER_URL']
    response = requests.post(url + "v1/models/sentiment_analysis:predict", json=request, headers=headers)
    if response.status_code != 200:
        return "error", 0.0

    result = response.json()
    predictions = result['predictions'][0]
    predictions = tf.nn.softmax(predictions).numpy()
    sentiment = 'NEGATIVE' if predictions[0] > predictions[1] else 'POSITIVE'
    confidence = max(predictions)
    return sentiment, confidence
