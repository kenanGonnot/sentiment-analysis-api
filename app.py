import os
from flask import Flask, jsonify, request
from sentiment_analysis import predict_sentiment_basic, predict_sentiment_tf_serving

app = Flask(__name__)


@app.route('/v1/inference/sentiment_analysis', methods=['POST'])
def sentiment_analysis_basic():
    """API call for Sentiment analysis V1 (basic).

    Returns: JSON:{
        sentiment: POSITIVE or NEGATIVE
        confidence: float(range(0, 1))
        }
    """
    data = request.get_json()

    if data is not None and 'sentence' in data:
        sentence = data['sentence']
        sentiment, confidence = predict_sentiment_basic(sentence)
        response_json = {
            'sentiment': sentiment,
            'confidence': round(float(confidence), 2)
        }
        return response_json
    else:
        return jsonify({'error': 'Invalid input'})


@app.route('/v2/inference/sentiment_analysis', methods=['POST'])
def sentiment_analysis_tf_serving():
    """API call for Sentiment analysis V2 (via TF-serving).

    Returns: JSON:{
        sentiment: POSITIVE or NEGATIVE
        confidence: float(range(0, 1))
        }
    """
    data = request.get_json()

    if data is not None and 'sentence' in data:
        sentence = data['sentence']
        sentiment, confidence = predict_sentiment_tf_serving(sentence)
        response_data = {
            'sentiment': sentiment,
            'confidence': round(float(confidence), 2)
        }
        return response_data
    else:
        return jsonify({'error': 'Invalid input'})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
