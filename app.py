from flask import Flask
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

app = Flask(__name__)
query_words = ['show', 'display', 'query', 'illustrate', 'print', 'select']
white_list_words = ['between']
num_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
lemmatizer = WordNetLemmatizer()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/tokenize/<sentence>', methods=['GET'])
def get_tokens(sentence):
    is_sql = False
    # Get Tokens
    tokens = word_tokenize(sentence)
    # Lemmatize Tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Pos-Tagging
    pos_tagged = nltk.pos_tag(lemmatized_tokens)
    single_word_tokens = []
    for word, tag in pos_tagged:
        if word in query_words and not is_sql:
            is_sql = True
            continue

        if tag not in ['DT', 'IN', 'PRP', 'CC']:
            if tag == 'CD':
                # Turn words to numbers for the lookup table
                if word in num_words:
                    word = str(num_words.index(word))

            single_word_tokens.append(word)
        elif word in white_list_words:
            single_word_tokens.append(word)

    bigrams = [' '.join(grams) for grams in ngrams(single_word_tokens, 2)]
    trigrams = [' '.join(grams) for grams in ngrams(single_word_tokens, 3)]

    response = {
        'query': sentence,
        'isSqlQuery': is_sql,
        'tokens': single_word_tokens,
        'bigrams': bigrams,
        'trigrams': trigrams
    }
    return response


if __name__ == '__main__':
    app.run()
