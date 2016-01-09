# -*- coding: utf-8 -*-
import os
from os import path
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel

def main():
    datadir = path.abspath(path.join(os.getcwd(), "data"))

    # load back the id->word mapping directly from file
    fin = path.join(datadir, "reuters21578.dict.txt")
    vocabulary = Dictionary.load_from_text(fin)

    # load the corpus
    fin = path.join(datadir, "reuters21578.mm")
    mm = MmCorpus(fin)

    # build tfidf, ~50min
    tfidf = TfidfModel(mm, id2word=vocabulary, normalize=True)

    # save the TfidfModel instance to file
    fout = path.join(datadir, "reuters21578.tfidf.model")
    tfidf.save(fout)

    # save TF-IDF vectors in matrix market format
    fout = path.join(datadir, "reuters21578.tfidf.mm")
    MmCorpus.serialize(fout, tfidf[mm], progress_cnt=10000)

if __name__ == "__main__":
    main()
