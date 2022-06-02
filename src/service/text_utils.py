from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from pymystem3 import Mystem
from string import punctuation
import cld2
from razdel import sentenize
# nltk.download("stopwords")

# Create lemmatizer and stopwords list
stemmer_dict = {"RUSSIAN": Mystem(),
              "ENGLISH": SnowballStemmer(language='english')}

stopwords_dict = {"RUSSIAN": stopwords.words("russian"),
                  "ENGLISH": stopwords.words("english")}


def detect_language(text_sample: str) -> str:
    """
    Return text_sample detected lang by cld2 library
    :param text_sample:
    :return: detected_lang as str
    """
    return cld2.detect(text_sample)[2][0][0]

# Preprocess function
def tokenize(text: str, text_lang: str) -> list:
    """
    Create tokenized stemmed text without punctuation and stopwords. Support 2 langs: Ru, Eng
    :param text: str
    :param text_lang: str
    :return: tokenized stemmed text without punctuation and stopwords
    """
    tokens = stemmer_dict[text_lang].lemmatize(text.lower()) if text_lang == "RUSSIAN" \
        else [stemmer_dict[text_lang].stem(token) for token in word_tokenize(text)]

    tokens = [token for token in tokens if token not in stopwords_dict[text_lang] \
              and token != " " \
              and token.strip() not in punctuation]

    # text = " ".join(tokens)

    return tokens


def custom_sentenize(text: str) -> list:
    """
    Split raw text with original formating saving
    :param text
    :return: list of raw text sents
    """
    sents = []
    for line in text.splitlines():
        line = line.lower() if line.strip().endswith("\n") else line
        sents += [sent.text for sent in sentenize(line) if sent.text != ""]

    # sent_char_lst = []
    # char_num = 0
    # for sent_num, sent in enumerate(sents):
    #     while char_num < len(text):
    #         if text[char_num:char_num + len(sent)] == sent:
    #             # Sentence found
    #             new_sent = text[char_num:char_num + len(sent)]
    #             sent_char_lst.append(new_sent)
    #             char_num += len(new_sent)
    #             break
    #         else:
    #             sent_char_lst.append(text[char_num])
    #             char_num += 1
    # # remaining characters
    # if char_num < len(text):
    #     sent_char_lst.append(text[char_num:])
    #
    # return [x for x in sent_char_lst if x != " "]


    return sents



# if __name__ == "__main__":
#     print("это" in stopwords_dict["RUSSIAN"])