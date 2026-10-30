# main.py
"""Interactive entry point for the original indexer/search logic."""

from pathlib import Path

from indexer import IndexBuilder
from search_engine import SearchEngine


class SearchEngineLauncher:
    
    def __init__(self):
        self.indexer = IndexBuilder()
        self.corpus_directory = None
        self.engine = None

    def run(self):
        try:
            # Prompt once and use the same corpus/database location for both actions.
            dirname = input('Enter corpus directory path: ').strip().strip('"')
            self.corpus_directory = Path(dirname).expanduser().resolve()
            self.engine = SearchEngine(self.corpus_directory / 'inverted_index.db')

            while True:
                print('\n1. Build Index')
                print('2. Search')
                print('3. Exit')
                choice = input('Choice: ').strip()

                if choice == '1':
                    self.build_index()
                elif choice == '2':
                    self.search()
                elif choice == '3':
                    break
                else:
                    print('Invalid choice')
        except (EOFError, KeyboardInterrupt):
            print('\nExiting.')

    def build_index(self):
        try:
            self.indexer.build_index(self.corpus_directory)
        except (ValueError, OSError) as e:
            print(f'Indexing error: {e}')

    def search(self):
        print('Search engine running...')
        try:
            # Prompt user for search terms
            query = input('Enter your search terms, each separated by a space: ')
            results = self.engine.search(query)
            print(f'The Top {min(len(results), 20)} Relevant Documents are:')
            for result in results:
                print(f'Document: {result.document_name} has Cosine Similarity of {result.cosine_similarity:.6f}')
            print('Search complete.')
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f'Search error: {e}')


if __name__ == '__main__':
    SearchEngineLauncher().run()
