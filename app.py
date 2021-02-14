from flask import Flask, request
import nltk
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from dateutil.relativedelta import *
import json

app = Flask(__name__)
white_list_words = ['between']
num_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
lemmatizer = WordNetLemmatizer()
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/tokenize/<sentence>', methods=['GET'])
def get_tokens(sentence):
    # Get Tokens
    tokens = word_tokenize(sentence)
    # Lemmatize Tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Pos-Tagging
    pos_tagged = nltk.pos_tag(lemmatized_tokens)
    single_word_tokens = []
    for word, tag in pos_tagged:
        # Filter unnecessary words
        if tag not in ['DT', 'IN', 'PRP', 'CC']:
            if tag == 'CD':
                # Turn words to numbers for the lookup table
                if word in num_words:
                    word = str(num_words.index(word))

            single_word_tokens.append(word)
        # Add whitelisted words to the list (e.g. between)
        elif word in white_list_words:
            single_word_tokens.append(word)

    bigrams = [' '.join(grams) for grams in ngrams(single_word_tokens, 2)]
    trigrams = [' '.join(grams) for grams in ngrams(single_word_tokens, 3)]

    response = {
        'query': sentence,
        'tokens': single_word_tokens,
        'bigrams': bigrams,
        'trigrams': trigrams
    }
    return response


@app.route('/predict', methods=['POST'])
def predict():
    num_predicted_months = 6
    date_format = '%Y-%m-%dT%H:%M:%S'
    sales = []
    dates = []
    data = request.json

    # Parse json request
    for row in data:
        sales.append(row['sales'])
        dates.append(datetime.strptime(row['date'], date_format))

    # Fit model
    model = SARIMAX(sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, len(sales))).fit()
    # Make prediction from last month to the next number of months
    forecast = model.predict(start=len(sales), end=len(sales) + num_predicted_months-1).tolist()

    # Map dates according to the forecast
    new_dates = [dates[-1]]
    for i in range(num_predicted_months):
        new_dates.append(new_dates[i] + relativedelta(months=+1))
    new_dates.pop(0)

    # Format json response
    forecast_dates = [datetime.strftime(date, date_format) for date in new_dates]
    output = json.dumps(
        [{'sales': sales, 'date': date} for sales, date in zip(forecast, forecast_dates)]
    )
    return output


if __name__ == '__main__':
    app.run()
