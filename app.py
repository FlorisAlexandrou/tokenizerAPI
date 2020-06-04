from flask import Flask, request
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/tokenize/<sentence>', methods=['GET'])
def get_tokens(sentence):
    # Get Tokens
    tokens = word_tokenize(sentence)
    # Lemmatize Tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Pos-Tagging
    pos_tagged = nltk.pos_tag(lemmatized_tokens)
    result = []
    for word, tag in pos_tagged:
        if tag not in ['DT', 'IN', 'PRP', 'CC']:
            result.append(word)

    response = {
        'query': sentence,
        'tokens': result
    }
    return response


if __name__ == '__main__':
    app.run()
