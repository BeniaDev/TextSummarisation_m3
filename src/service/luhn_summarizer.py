from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from razdel import sentenize
from nltk.corpus import stopwords
from text_utils import tokenize
import numpy as np
# download stopwords corpus, you need to run it once
# nltk.download("stopwords")


from pymystem3 import Mystem

#Create lemmatizer and stopwords list
mystem = Mystem()
russian_stopwords = stopwords.words("russian")


class LuhnSummarizer():
    def __init__(self, is_lemmatize: bool = True):
        self.is_lemmatize = True
        self.sf_word_threshold = 0.002
        self.sf_sentence_threshold = 0.3
        self.max_not_sf_seq_words = 4

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
        vectorizer = TfidfVectorizer(use_idf=False)
        X = vectorizer.fit_transform(tokens)
        features_names_out = vectorizer.get_feature_names_out()

        freq_df = pd.DataFrame(X.toarray(), columns=features_names_out)  # составляем датафрейм частотностей слов
        # print(f"Features Names Out: {features_names_out}")

        freq_dict = {word: np.sum(freq_df[word].values) / len(list(freq_df.keys())) for word in features_names_out}

        freq_dict = dict(sorted(freq_dict.items(), key=lambda y: y[1],
                                reverse=True))  # сортируем словарь частотностей слов по убыванию
        # print(f"Freq Dict before split by threshold: {freq_dict}")
        freq_dict = {k: v for k, v in freq_dict.items() if v >= self.sf_word_threshold}

        return freq_dict

    def get_sentence_significance_word_mask(self, sentence_words_mask: list) -> list:
        first_sf_word_indx = sentence_words_mask.index(1)
        last_sf_word_indx = len(sentence_words_mask) - 1 - sentence_words_mask[::-1].index(1)

        return sentence_words_mask[first_sf_word_indx: last_sf_word_indx + 1]

    def compute_significance_factor(self, freq_dict: dict, sentence: list) -> np.float16:
        sentence_words_mask = [1 if freq_dict[word] else 0 \
                               for word in sentence if word in freq_dict.keys()]
        # example of sentence_words_mask: [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0] 1 - sf word in sentence and 0 - not sf word
        # print(sentence_words_mask)

        if sum(sentence_words_mask) == 0:
            return 0

        sentence_words_mask = self.get_sentence_significance_word_mask(sentence_words_mask)
        number_of_sf_words = sum(sentence_words_mask)
        total_number_of_bracketed_words = len(sentence_words_mask)

        # print(f"number_of_sf_words: {number_of_sf_words}, total_number_of_bracketed_words: {total_number_of_bracketed_words}")
        significance_factor = number_of_sf_words ** 2 / total_number_of_bracketed_words

        return significance_factor

    def summarize(self, text: str) -> str:
        # sentence segmentation
        sentences = self.custom_sentenize(text)
        # print(len(sentences))
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

        sentence_sf_threshold_percentile_75 = np.percentile(sentences_sf, 75)
        # print(sentences_sf)
        # print(sentences_sf[ranking_ind_uppper_bound])
        # log(min(sentences_sf[:ranking_ind_uppper_bound]))

        summary_sentences = [sent for i, sent in enumerate(sentences) if
                             sentences_sf[i] >= sentence_sf_threshold_percentile_75]

        return "".join(summary_sentences)