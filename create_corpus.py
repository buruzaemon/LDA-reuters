# -*- coding: utf-8 -*-
import os
from os import path
from eda.corpus.reuterscorpus import ReutersCorpus
from gensim.corpora import MmCorpus

def main():
    datadir = path.abspath(path.join(os.getcwd(), "data"))

    # Read in the corpus from within the archive file
    fin = path.join(datadir, "reuters21578.tar.gz")
    rc = ReutersCorpus(fin)

    # filter out some of the more common words,
    # and some of the less-common ones as well
    rc.dictionary.filter_extremes(no_below=20, no_above=0.1)
    rc.dictionary.compactify()

    # Serialize the Reuters 21578 corpus
    fout = path.join(datadir, "reuters21578.mm")
    MmCorpus.serialize(fout, rc)

    # Save the dictionary to file as text
    fout = path.join(datadir, "reuters21578.dict.txt")
    rc.dictionary.save_as_text(fout)

if __name__ == "__main__":
    main()
