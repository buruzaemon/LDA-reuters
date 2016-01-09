# -*- coding: utf-8 -*-
"""
Simple implementation of Gensim TextCorpus.

Reads the *.sgm files within reuters21578.tar.gz
"""
import tarfile
from bs4 import BeautifulSoup
from eda.text.utilities import clean, tokenize
from contextlib import closing
# cannot import whole gensim.corpora, because that imports wikicorpus...
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus


class ReutersCorpus(TextCorpus):
    """
    Treat the Reuters 21578 dataset (data/reuters21578.tar.gz) as a
    read-only corpus.

    The documents are read in while iterating the archive file.

    >>> rc = ReutersCorpus('data/reuters21578.tar.gz')
    >>> MmCorpus.serialize('data/reuters_bow.mm', rc)

    """
    def __init__(self, fname, dictionary=None):
        """
        Initialize the corpus. Unless a dictionary is provided, this scans the
        corpus once, to determine its vocabulary.
        """
        self.fname = fname
        self.metadata = False

        if dictionary is None:
            dictionary = Dictionary()
            for text in self.get_texts():
                dictionary.add_documents([text])
        self.dictionary = dictionary


    def get_texts(self):
        """
        Iterate over the collection, yielding one document at a time.
        A document is a sequence of words (strings) that can be fed into
        `Dictionary.doc2bow`.
        
        A document will include:
        - <TITLE>
        - <DATELINE>
        - <BODY> (if available)
        """
        with tarfile.open(self.fname, mode="r") as archive:
            for f in archive:
                if f.isreg() and f.name.endswith(".sgm"):
                    with closing(archive.extractfile(f)) as data:
                        soup = BeautifulSoup(data, "html.parser")
                        for article in soup.find_all("reuters"):
                            text = []
                            if article.title:
                                text.append(article.title.text.strip())
                            if article.dateline:
                                text.append(article.dateline.text.strip())
                            if article.body:
                                text.append(article.body.text.strip())

                            yield tokenize(clean(" ".join(text)))

# endclass ReutersCorpus
