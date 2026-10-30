"""Corpus token selection, stopword filtering, and Porter stemming."""

import re
import string
from dataclasses import dataclass

from porter_stemmer import PorterStemmer
from stopwords import mixed_source_stopwords


@dataclass
class ProcessingResult:
    terms: list[str]
    tokens: int
    stop_words_count: int


class TextProcessor:

    # Regular expression to extract words, extract ID from path, check for hexa value
    chars = re.compile(r'\W+')
    pattid = re.compile(r'(\d{3})/(\d{3})/(\d{3})')

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = mixed_source_stopwords

    # split on any chars
    def splitchars(self, line):
        return self.chars.split(line)

    # A Function that removes "stopwords" from a "list" and applies stemming
    def remove_stopwords_and_stem(self, tokens, stopwords=None):
        processed_list = []
        stop_words_count = 0
        if stopwords is None:
            stopwords = self.stopwords
        stem = self.stemmer
        for word in tokens:
            if word not in stopwords:
                processed_list.append(stem.stem(word, 0, len(word) - 1))
            else:
                stop_words_count += 1
        return processed_list, stop_words_count

    def process_document(self, data):
        # For each document in the directory read the document into a lower case normalized string
        # after removing tokens beginning with punctuation
        alltokens = []
        data = data.replace('\n', ' ').lower()
        for token in data.split():
            if token[0] not in string.punctuation and not token.isdigit() and not len(token) < 2:
                alltokens.append(token)
        processed, stop_words_count = self.remove_stopwords_and_stem(alltokens)
        return ProcessingResult(processed, len(alltokens), stop_words_count)

    def process_query(self, query):
        
        l = self.splitchars(query)
        words = [word.lower().strip() for word in l if word.strip()]
        processed, _ = self.remove_stopwords_and_stem(words)
        return processed
