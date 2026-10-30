"""Build the original term/document dictionaries, postings and TF-IDF index."""

import os
import time
from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path

from database import SQLiteIndex
from text_processing import TextProcessor


@dataclass
class IndexStats:
    documents: int
    tokens: int
    terms: int
    stop_words_count: int
    db_path: Path


class IndexBuilder:
    """Index a corpus without modifying the original text-processing or TF-IDF formula."""

    # Generated outputs must not be indexed as new corpus documents on a rebuild.
    GENERATED_FILES = {
        'documents.dat', 'document dictionary.dat', 'index.dat',
        'term dictionary.dat', 'postings.dat', 'inverted_index.db',
    }

    def __init__(self, processor=None):
        self.processor = processor if processor is not None else TextProcessor()

    def build_index(self, dirname):
        # Define global variables used as counters
        tokens = 0
        documents = 0
        terms = 0
        termindex = 0
        docindex = 0
        stop_words_count = 0

        # Capture the start time of the routine so that we can determine the total running
        # time required to process the corpus
        t2 = time.localtime()

        # Set the name of the directory for the corpus
        directory = Path(dirname).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError('Error: Directory does not exist.')

        # Initialize list variables
        alltokens = []
        alldocs = []
        document_terms = {}

        # For each document in the directory read the document into a lower case normalized string
        # after removing tokens beginning with punctuation
        # As in the original, only files directly inside the selected directory are indexed.
        all_files = sorted(f for f in os.listdir(directory)
                           if (directory / f).is_file()
                           and f.lower() not in self.GENERATED_FILES)
        for f in all_files:
            # Avoid re-indexing a custom database path if one is located here.
            if (directory / f).suffix.lower() in ('.db', '.sqlite', '.sqlite3'):
                continue
            try:
                with open(directory / f, 'r', encoding='utf-8') as myfile:
                    data = myfile.read()
            except (OSError, UnicodeError):
                # A directory can contain non-text files; only readable documents count.
                continue
            result = self.processor.process_document(data)
            documents += 1
            alldocs.append(f)
            document_terms[f] = Counter(result.terms)
            alltokens.extend(result.terms)
            tokens += result.tokens
            stop_words_count += result.stop_words_count

        if documents == 0:
            raise ValueError('No readable documents in the corpus directory.')

        # Removing stopwords from all tokens
        # This happened per document above.

        # Open for write a file for the document dictionary
        document_rows = []
        alldocs.sort()
        for f in alldocs:
            docindex += 1
            document_rows.append((docindex, f))

        with open(directory / 'documents.dat', 'w', encoding='utf-8') as documentfile:
            for docid, f in document_rows:
                documentfile.write(f + ',' + str(docid) + os.linesep)

        # Open for write a file for the document dictionary
        with open(directory / 'Document Dictionary.dat', 'w', encoding='utf-8') as documentfile:
            for docid, f in document_rows:
                documentfile.write(f + ',' + str(docid) + os.linesep)

        # Sort the tokens in the list
        alltokens.sort()

        # Define a list for the unique terms
        g = []

        # Identify unique terms in the corpus
        g = sorted(set(alltokens))
        terms = len(g)

        # Output Index to disk file. As part of this process we assign an 'index' number to each unique term.
        term_ids = {}
        with open(directory / 'index.dat', 'w', encoding='utf-8') as indexfile:
            for i in g:
                termindex += 1
                term_ids[i] = termindex
                indexfile.write(i + ',' + str(termindex) + os.linesep)

        # Output Index to disk file. As part of this process we assign an 'index' number to each unique term.
        with open(directory / 'Term Dictionary.dat', 'w', encoding='utf-8') as indexfile:
            for i in g:
                indexfile.write(i + ',' + str(term_ids[i]) + os.linesep)

        # Calculate term frequency (TF), document frequency (DF), and inverse document frequency (IDF)
        tf = Counter(alltokens)  # Original: total frequency over all processed corpus tokens.
        df = {}
        for term in g:
            df[term] = sum(1 for doc in alldocs if term in document_terms[doc])

        idf = {term: log(documents / (df[term] + 1)) for term in g}

        # Calculate tf-idf weighting
        tf_idf = {term: tf[term] * idf[term] for term in g}

        # Output postings to disk file
        posting_rows = []
        with open(directory / 'Postings.dat', 'w', encoding='utf-8') as postingsfile:
            for term in g:
                term_id = term_ids[term]
                for doc_id, doc in document_rows:
                    term_freq = document_terms[doc][term]
                    if not term_freq:
                        # An inverted index only includes documents with this term.
                        continue
                    doc_freq = df[term]
                    tf_idf_weight = tf_idf[term]  # Keep the original corpus-wide TF-IDF.
                    postingsfile.write(
                        f'{term_id},{doc_id},{tf_idf_weight},{term_freq},{doc_freq}{os.linesep}'
                    )
                    posting_rows.append((term_id, doc_id, tf_idf_weight, term_freq))

        # Creates database
        # build real SQLite tables so Search_Engine can read them.
        term_rows = [(term_ids[term], term, df[term], tf[term], idf[term]) for term in g]
        storage = SQLiteIndex(directory / 'inverted_index.db')
        storage.save_index(document_rows, term_rows, posting_rows)

        # Print metrics on corpus
        print(f'Processing Start Time: {t2.tm_hour:02d}:{t2.tm_min:02d}:{t2.tm_sec:02d}')
        print(f'Documents: {documents}')
        print(f'Tokens: {tokens}')
        print(f'Terms: {terms}')
        print(f'Stop words encountered: {stop_words_count}')
        t2 = time.localtime()
        print(f'Processing End Time: {t2.tm_hour:02d}:{t2.tm_min:02d}:{t2.tm_sec:02d}')
        print('Index built successfully.')
        return IndexStats(documents, tokens, terms, stop_words_count, storage.db_path)
