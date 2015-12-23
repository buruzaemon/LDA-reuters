# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:24:53 2015

@author: buruzaemon
"""
import nltk
import re
import tarfile
from bs4 import BeautifulSoup
from contextlib import closing
from nltk.corpus import stopwords


# cannot import whole gensim.corpora, because that imports wikicorpus...
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus


PATT = r'''(?x)
# abbreviations, e.g. U.S.A.
([A-Z]\.)+

# major cities / countries, attempt #1
| (?<![a-zA-Z])((Buenos|Cape|East(ern)?|El|Fort|Hong|Las|Los|New|North(ern)?|Puerto|Port|Rio(\sde)?|Saint|San|St\.|São|Sierra|Saudi|South(ern)?|United|West(ern)?)[\s][A-Z][a-z]+)(?![a-zA-Z])

# major cities / countries, attempt #2
| (?<![a-zA-Z])(([A-Z][a-z]+)\s(City|Island(s)|Republic))(?![a-zA-Z])

# words with optional internal hyphens
| \w+(-\w+)*        

# currency and percentages, e.g. $12.40, 82%
| \$?\d+(\.\d+)?%?  

# ellipsis
| \.\.\. 

# these are separate tokens; includes ], [           
| [][.,;"'?!():-_`]  
'''


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
        
    def clean_text(self, text):
        # Gensim Dictionary is tab-delimited, 
        # so we cannot allow allow any tabs in the text
        cleaned = re.sub(r'\t', r' ', text)
        
        cleaned = re.sub(r'\n', r' ', cleaned)
        
        return cleaned

    def get_texts(self):
        """
        Iterate over the collection, yielding one document at a time. 
        A document is a sequence of words (strings) that can be fed into
        `Dictionary.doc2bow`.
        """
        ignore = stopwords.words("english")
        lemm = nltk.WordNetLemmatizer()
        
        with tarfile.open(self.fname, mode="r") as archive:
            for f in archive:
                if f.isreg() and f.name.endswith(".sgm"):
                    with closing(archive.extractfile(f)) as data:
                        soup = BeautifulSoup(data, "html.parser")
                        for body in soup.find_all("body"):
                            text = self.clean_text(body.text)
                            tokens = [t.lower() for t in nltk.regexp_tokenize(text, PATT)]
        
                            # HOW DO YOU WANT TO FILTER THE TEXT TOKENS?
                            yield [lemm.lemmatize(t) for t in tokens if t not in ignore]

# endclass ReutersCorpus

