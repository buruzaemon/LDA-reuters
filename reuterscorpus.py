# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:24:53 2015

@author: buruzaemon
"""
import nltk
import tarfile
from bs4 import BeautifulSoup
from contextlib import closing
from nltk.corpus import stopwords


# cannot import whole gensim.corpora, because that imports wikicorpus...
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus


PATT = r'''(?x)    # set flag to allow verbose regexps
    ([A-Z]\.)+          # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*        # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    | \.\.\.            # ellipsis
    | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
'''


class ReutersCorpus(TextCorpus):
    """
    Treat a wikipedia articles dump (\*articles.xml.bz2) as a (read-only) corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.

    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id mapping, takes almost 8h
    >>> MmCorpus.serialize('wiki_en_vocab200k', wiki) # another 8h, creates a file in MatrixMarket format plus file with id->word

    """
    def __init__(self, fname, dictionary=None):
        """
        Initialize the corpus. Unless a dictionary is provided, this scans the
        corpus once, to determine its vocabulary.

        If `pattern` package is installed, use fancier shallow parsing to get
        token lemmas. Otherwise, use simple regexp tokenization. You can override
        this automatic logic by forcing the `lemmatize` parameter explicitly.

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
        Iterate over the collection, yielding one document at a time. A document
        is a sequence of words (strings) that can be fed into `Dictionary.doc2bow`.

        Override this function to match your input (parse input files, do any
        text preprocessing, lowercasing, tokenizing etc.). There will be no further
        preprocessing of the words coming out of this function.
        """
        ignore = stopwords.words("english")
        lemm = nltk.WordNetLemmatizer()
        
        with tarfile.open(self.fname, mode="r") as archive:
            for f in archive:
                if f.isreg() and f.name.endswith(".sgm"):
                    with closing(archive.extractfile(f)) as data:
                        soup = BeautifulSoup(data, "html.parser")
                        for body in soup.find_all("body"):
                            tokens = nltk.regexp_tokenize(body.text, PATT)
        
                            # HOW DO YOU WANT TO FILTER THE TEXT TOKENS?
                            yield [lemm.lemmatize(t.lower()) for t in tokens if t not in ignore]

# endclass ReutersCorpus

