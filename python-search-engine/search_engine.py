"""
A search engine that will search an inverted index
"""

import math
import time
from collections import Counter
from dataclasses import dataclass

from database import SQLiteIndex
from text_processing import TextProcessor


#
# Docs class: Used to store information about each unit document. In this is the Term object which stores each
# unique instance of termid or a docid.
#
class Docs:
    def __init__(self):
        self.terms = {}


#
# Term class: used to store information or each unique termid
#
class Term:
    def __init__(self):
        self.docfreq = 0
        self.termfreq = 0
        self.idf = 0.0
        self.tfidf = 0.0


@dataclass(frozen=True)
class SearchResult:
    docid: int
    document_name: str
    cosine_similarity: float


class SearchEngine:

    def __init__(self, db_path, processor=None):
        self.storage = SQLiteIndex(db_path)
        self.processor = processor if processor is not None else TextProcessor()

    # split on any chars
    def splitchars(self, line):
        return self.processor.splitchars(line)

    # this small routine is used to accumulate query idf values
    def elenQ(self, elen, term):
        return(float(math.pow(term.idf, 2)) + float(elen))

    # this small routine is used to accumulate document tfidf values
    def elenD(self, elen, term):
        return(float(math.pow(term.tfidf, 2)) + float(elen))

    # Function to get the total number of documents in the collection
    def get_document_count(self, cur):
        return self.storage.get_document_count(cur)

    # Helper function that checks whether a given term exists in the termdictionary table of the database.
    def term_in_database(self, term, cur):
        return self.storage.term_in_database(term, cur)

    # Function to process the query and create a query vector
    def process_query(self, query, cur):
        l = self.processor.process_query(query)
        query_terms = Counter(l) #count the occurrences of each element in a list, l
        query_vector = {}
        for term, freq in query_terms.items():
            lower_term = term.lower().strip()
            query_vector[lower_term] = freq

        # Validate terms against the database
        query_vector = {term: freq for term, freq in query_vector.items()
                        if self.term_in_database(term, cur)}
        return query_vector

    # Function to fetch document data for the query terms
    def fetch_document_data(self, query_vector, cur, docs, documents):
        query_term_ids = {}
        for term in query_vector.keys():
            termid = self.storage.get_term_id(term, cur)
            if termid is None:
                continue
            query_term_ids[term] = termid
            for row in self.storage.get_postings(term, cur):
                i_termid = row[4]
                i_docid = row[0]

                if i_docid not in docs:
                    docs[i_docid] = Docs()

                if i_termid not in docs[i_docid].terms:
                    docs[i_docid].terms[i_termid] = Term()
                    docs[i_docid].terms[i_termid].docfreq = row[2]
                    docs[i_docid].terms[i_termid].termfreq = row[3]
                    docs[i_docid].terms[i_termid].idf = math.log(documents / (row[2] + 1))
                    docs[i_docid].terms[i_termid].tfidf = row[1]
        return query_term_ids

    # Function call to compute cosine similarity between the query and each document
    def compute_similarities(self, query_vector, query_term_ids, docs, documents):
        resultslist = {}
        #
        #Computing the Euclidean length (or magnitude) of the query vector
        #
        query_length = 0
        for term, termid in query_term_ids.items():
            for docid in docs.keys():
                if termid in docs[docid].terms:
                    query_term = Term() # Setting the idf value for the term
                    query_term.idf = math.log(documents / (docs[docid].terms[termid].docfreq + 1))
                    query_length = self.elenQ(query_length, query_term)
                    break  # Count each query term once, not once per matching doc.
        query_length = math.sqrt(query_length)

        for docid in docs.keys():
            doc_vector = docs[docid].terms
            dot_product = 0
            for term, termid in query_term_ids.items():
                if termid in doc_vector:
                    dot_product += query_vector[term] * doc_vector[termid].tfidf

            doc_length = 0
            for termid in doc_vector:
                doc_length += self.elenD(0, doc_vector[termid])
            doc_length = math.sqrt(doc_length)

            if query_length > 0 and doc_length > 0:
                cosine_similarity = dot_product / (query_length * doc_length)
                resultslist[docid] = cosine_similarity 

        return resultslist

    def search(self, query, limit=20):
        if not self.storage.has_index():
            raise FileNotFoundError('No index found. Choose 1 → Build Index before searching.')

        # Record start time
        t_start = time.time()
        # Use simple dictionary data structures in Python to maintain lists with hash keys
        docs = {}
        con, cur = self.storage.initialize_database()
        try:
            # Get the total number of documents
            documents = self.get_document_count(cur)

            # Process query
            query_vector = self.process_query(query, cur)
            if not query_vector:
                return []

            # Fetch document data
            query_term_ids = self.fetch_document_data(query_vector, cur, docs, documents)

            # Remove documents not containing all query terms
            for docid in list(docs.keys()):
                if not all(termid in docs[docid].terms for termid in query_term_ids.values()):     # Check if all query terms are present in the document's terms
                    del docs[docid]

            # Compute similarities
            resultslist = self.compute_similarities(query_vector, query_term_ids, docs, documents)

            # Sort and print the top 20 results
            keylist = sorted(resultslist, key=lambda docid: (-resultslist[docid], docid))
            return [SearchResult(docid, self.storage.get_document_name(docid, cur), resultslist[docid])
                    for docid in keylist[:limit]]
        finally:
            con.close()
            # Print elapsed time
            t_end = time.time()
            print(f'Search completed in {t_end - t_start:.2f} seconds.')
