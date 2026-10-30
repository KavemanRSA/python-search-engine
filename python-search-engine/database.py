"""SQLite persistence for the original term dictionary, document dictionary and postings."""

import sqlite3
from pathlib import Path


class SQLiteIndex:
    """Owns only the database connection, schema and simple SQL access."""

    def __init__(self, db_path):
        self.db_path = Path(db_path).expanduser().resolve()

    # Function to initialize database connection
    def initialize_database(self):
        try:
            con = sqlite3.connect(self.db_path)
            con.isolation_level = None
            cur = con.cursor()
            return con, cur
        except sqlite3.Error as e:
            raise RuntimeError(f"Database connection error: {e}") from e

    def has_index(self):
        # sqlite3.connect creates a nonexistent file: check before opening for search.
        if not self.db_path.is_file():
            return False
        try:
            with sqlite3.connect(self.db_path) as con:
                names = {row[0] for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )}
            return {'documentdictionary', 'termdictionary', 'posting'} <= names
        except sqlite3.Error:
            return False

    # Creates database
    def save_index(self, document_rows, term_rows, posting_rows):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path)
        try:
            with con:
                con.execute('DROP TABLE IF EXISTS posting')
                con.execute('DROP TABLE IF EXISTS termdictionary')
                con.execute('DROP TABLE IF EXISTS documentdictionary')
                con.execute('''CREATE TABLE documentdictionary (
                    docid INTEGER PRIMARY KEY,
                    DocumentName TEXT NOT NULL UNIQUE
                )''')
                con.execute('''CREATE TABLE termdictionary (
                    termid INTEGER PRIMARY KEY,
                    term TEXT NOT NULL UNIQUE,
                    docfreq INTEGER NOT NULL,
                    termfreq INTEGER NOT NULL,
                    idf REAL NOT NULL
                )''')
                con.execute('''CREATE TABLE posting (
                    termid INTEGER NOT NULL,
                    docid INTEGER NOT NULL,
                    tfidf REAL NOT NULL,
                    termfreq INTEGER NOT NULL,
                    PRIMARY KEY (termid, docid),
                    FOREIGN KEY (termid) REFERENCES termdictionary(termid),
                    FOREIGN KEY (docid) REFERENCES documentdictionary(docid)
                )''')
                con.executemany(
                    'INSERT INTO documentdictionary (docid, DocumentName) VALUES (?, ?)',
                    document_rows,
                )
                con.executemany(
                    'INSERT INTO termdictionary (termid, term, docfreq, termfreq, idf) '
                    'VALUES (?, ?, ?, ?, ?)', term_rows,
                )
                con.executemany(
                    'INSERT INTO posting (termid, docid, tfidf, termfreq) '
                    'VALUES (?, ?, ?, ?)', posting_rows,
                )
                con.execute('CREATE INDEX IF NOT EXISTS idx_posting_docid ON posting(docid)')
        finally:
            con.close()

    # Function to get the total number of documents in the collection
    def get_document_count(self, cur):
        q = 'select count(*) from documentdictionary'
        cur.execute(q)
        row = cur.fetchone()
        return row[0]

    # Helper function that checks whether a given term exists in the termdictionary table of the database.
    def term_in_database(self, term, cur):
        """
        Checks if a given term exists in the termdictionary table of the database.

        Args:
            term (str): The term to check.
            cur (sqlite3.Cursor): The database cursor object.

        Returns:
            bool: True if the term exists, False otherwise.
        """
        # SQL query to count occurrences of the term in the database (case-insensitive)
        q = 'SELECT COUNT(*) FROM termdictionary WHERE LOWER(term) = ?'
        cur.execute(q, (term.lower(),))  # Use the term in lowercase for consistent matching
        row = cur.fetchone()
        return row[0] > 0  # If count > 0, the term exists in the database

    # Function to fetch document data for the query terms
    def get_postings(self, term, cur):
        q = '''
            SELECT p.docid, p.tfidf, t.docfreq, p.termfreq, p.termid
            FROM termdictionary AS t
            JOIN posting AS p ON p.termid = t.termid
            WHERE t.term = ?
            ORDER BY p.docid, p.termid
        '''
        return cur.execute(q, (term,)).fetchall()

    def get_term_id(self, term, cur):
        row = cur.execute('SELECT termid FROM termdictionary WHERE term = ?', (term,)).fetchone()
        return None if row is None else row[0]

    def get_document_name(self, docid, cur):
        row = cur.execute(
            'SELECT DocumentName FROM documentdictionary WHERE docid = ?', (docid,)
        ).fetchone()
        return None if row is None else row[0]
