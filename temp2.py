# -*- coding: utf-8 -*-
"""
pyLDAvis initialization/execution
"""
import os
import pyLDAvis.gensim
import socket
import sys

from os import path
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel

def main():
    datadir = path.abspath(path.join(os.getcwd(), "data"))

    # load the LDA model
    fin = path.join(datadir, "reuters21578.lda.model.bz2")
    lda = LdaModel.load(fin)

    # load the corpus
    fin = path.join(datadir, "reuters21578.mm.bz2")
    mm = MmCorpus(fin)


    # load the vocabulary
    fin = path.join(datadir, "reuters21578.dict.txt")
    vocabulary = Dictionary.load_from_text(fin)

    data = pyLDAvis.gensim.prepare(lda, corpus, vocabulary)
    pyLDAvis.show(data,
                  ip=socket.gethostname().lower(),
                  local=True,
                  open_browser=True,
                  http_server=None)

if __name__ == "__main__":
    main()
