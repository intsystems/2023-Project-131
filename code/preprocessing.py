import numpy as np
import pandas as pd

# for preprocessing
import nltk

from nltk.collocations import (
    BigramAssocMeasures,
    BigramCollocationFinder,
)
from nltk.corpus import (
    stopwords,
    wordnet,
)

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_ru')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer

from pymorphy3 import MorphAnalyzer

from collections import Counter

from tqdm import tqdm


def vowpalize_sequence(sequence):
    '''
    Переводит последовательность в формат из Vowpal Wabbit.

    Аргумент:
    sequence - последовательность

    Возвращает:
    res - результат преобразования
    '''
    word_freq = Counter(sequence)
    del word_freq['']
    
    res = ''
    for word in word_freq:
        res += word + ":" + str(word_freq[word]) + ' '
  
    return res


def process_data(dataframe, vocab_size=16000):
    '''
    Отвечает за предобработку набора данных для
    дальнейшего использования в TopicNet.

    Аргумент:
    dataframe - данные, содержащие колонку raw_text
    vocab_size - размер словаря биграмм

    Результат:
    dataframe - предобработанные данные
    '''

    tokenized_text = []  # инициализируем список
    
    # разбиваем на токены
    for _, data in tqdm(dataframe.iterrows()):
        tokens = [token for token in nltk.wordpunct_tokenize(data.title.lower()) if len(token) > 1]
        tokens.extend([token for token in nltk.wordpunct_tokenize(data.raw_text.lower()) if len(token) > 1])
        tokenized_text.append(tokens)
    
    # запоминаем токенизацию
    dataframe['tokenized'] = tokenized_text

    # список стопслов
    stop = set(stopwords.words('russian'))

    # применяем лемматизацию, используем PyMorphy3
    lemmatized_text = []
    morph = MorphAnalyzer()
    
    # переводим в нормальную форму
    for text in tqdm(dataframe['tokenized'].values):
        lemmatized = [morph.parse(word)[0].normal_form for word in text]
        lemmatized = [word for word in lemmatized 
                      if word not in stop and word.isalpha()]
        lemmatized_text.append(lemmatized)
    
    dataframe['lemmatized'] = lemmatized_text

    # выбираем лучшие биграммы
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents(dataframe['lemmatized'])
    finder.apply_freq_filter(5)
    set_dict = set(finder.nbest(bigram_measures.pmi, vocab_size))
    documents = dataframe['lemmatized']
    bigrams = []

    # добавляем биграммы
    for doc in tqdm(documents):
        entry = ['_'.join([word_first, word_second])
                 for word_first, word_second in zip(doc[:-1],doc[1:])
                 if (word_first, word_second) in set_dict]
        bigrams.append(entry)

    dataframe['bigram'] = bigrams
    
    # добавялем текст в формате из Vowpal Wabbit
    vw_text = []

    for index, data in tqdm(dataframe.iterrows()):
        vw_string = ''    
        doc_id = str(index)
        lemmatized = '@lemmatized ' + vowpalize_sequence(data.lemmatized)
        bigram = '@bigram ' + vowpalize_sequence(data.bigram)
        vw_string = ' |'.join([doc_id, lemmatized, bigram])
        vw_text.append(vw_string)

    dataframe['vw_text'] = vw_text

    return dataframe