from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

# Create lemmatizer and stopwords list
mystem = Mystem()
russian_stopwords = stopwords.words("russian")


# Preprocess function
def tokenize(text: str) -> list:
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and token.strip() not in punctuation]

    # text = " ".join(tokens)

    return tokens