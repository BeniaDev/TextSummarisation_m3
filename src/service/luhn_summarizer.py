from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from razdel import sentenize
from nltk.corpus import stopwords
from text_utils import tokenize
# download stopwords corpus, you need to run it once
# nltk.download("stopwords")


from pymystem3 import Mystem

#Create lemmatizer and stopwords list
mystem = Mystem()
russian_stopwords = stopwords.words("russian")

class LuhnSummarizer():
    def __init__(self, is_lemmatize: bool = True):
        self.is_lemmatize = True
        self.sf_word_threshold = 0.25

    def custom_sentenize(self, text: str) -> list:
        sents = []
        for line in text.splitlines():
            line = line.lower() if line.strip().endswith("\n") else line
            sents += [sent.text for sent in sentenize(line) if sent.text != ""]

        sent_char_lst = []
        char_num = 0
        for sent_num, sent in enumerate(sents):
            while char_num < len(text):
                if text[char_num:char_num + len(sent)] == sent:
                    # Sentence found
                    new_sent = text[char_num:char_num + len(sent)]
                    sent_char_lst.append(new_sent)
                    char_num += len(new_sent)
                    break
                else:
                    sent_char_lst.append(text[char_num])
                    char_num += 1
        # remaining characters
        if char_num < len(text):
            sent_char_lst.append(text[char_num:])

        return [x for x in sent_char_lst if x != " "]

    def tokenize_sent(self, sentences: list) -> list:

        # tokenization
        tokens = [tokenize(sent) for sent in sentences]  # [x for sent in sentences for x in tokenize(sent)]

        return tokens

    def create_word_freq_dict(self, text: str) -> dict:
        # log(f"tokens:{tokens}")

        tokens = tokenize(text)
        vectorizer = TfidfVectorizer()  # (use_idf=False)
        X = vectorizer.fit_transform(tokens)
        features_names_out = vectorizer.get_feature_names_out()

        freq_df = pd.DataFrame(X.toarray(), columns=features_names_out)  # составляем датафрейм частотностей слов
        freq_dict = {word: np.sum(freq_df[word].values) for word in features_names_out}

        return freq_dict

    def compute_significance_factor(self, freq_dict: dict, sentence: list) -> np.float16:

        all_words_count = len(sentence)
        number_of_sf_words = sum(1 if freq_dict[word] / all_words_count > self.sf_word_threshold else 0 \
                                 for word in sentence if word in freq_dict.keys())

        significance_factor = number_of_sf_words ** 2 / all_words_count

        return significance_factor

    def get_sentence_significance_word_cluster(self, freq_dict: dict, text: str) -> str:
        pass

    def summarize(self, text: str) -> str:
        # sentence segmentation
        sentences = self.custom_sentenize(text)
        # log(sentences)

        text_freq_dict = self.create_word_freq_dict(text)
        # log(text_freq_dict)

        sentences_sf = []
        for sent in sentences:
            sentence_tokens = tokenize(sent)
            # log(sentence_tokens)

            sentences_sf.append(
                self.compute_significance_factor(text_freq_dict, sentence_tokens) if len(sentence_tokens) > 0 else 0)
        # log(sentences_sf)

        sentences_sf.sort(reverse=True)

        ranking_ind_uppper_bound = int(len(sentences_sf) * 0.05) if len(sentences_sf) * 0.05 > 1 else 1
        # log(min(sentences_sf[:ranking_ind_uppper_bound]))

        summary_sentences = [sent for i, sent in enumerate(sentences) if
                             sentences_sf[i] >= sentences_sf[ranking_ind_uppper_bound]]

        return summary_sentences