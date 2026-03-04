# python-search-engine

A basic search engine in Python using TF-IDF indexing, stopword removal, and stemming. 
It builds a term dictionary and a document dictionary and supports querying over a document corpus.

I approached this search engine project by starting at the highest level, by defining the system as two core components: an index builder and a query processor. 
From there, I decomposed the indexer into stages: corpus ingestion, normalisation, stopword removal and stemming, term dictionary creation, postings generation, and TF-IDF weighting. 
Separately, I structured the search engine into query processing, database retrieval, vector construction, and cosine similarity ranking. By designing the architecture first and then 
implementing each module with clear responsibilities, I ensured the system remained modular, scalable, and aligned with information retrieval theory.

Features:
- Tokenization
- Stopword removal
- Porter stemming
- Term dictionary
- Document dictionary
- TF-IDF weighting
- Search functionality

Built using:
- Python 3
- File handling
- Data structures (lists, dictionaries)
- Information retrieval concepts

How to run:
1. Run launcher.py
2. Enter the corpus directory when prompted
3. For 1st time usage, select:
    1 → Build index
  then:
    2 → Search
  Else, having run before, select:
    2 → Search
